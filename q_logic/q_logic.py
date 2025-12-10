
import json
import sys
import random
import time
import traceback
import copy
import os

import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
import json
import torch.nn.functional as F


from collections import deque
from q_logic.loss_functions import huberLoss
from q_logic.q_logic_memory_classes import ReplayBuffer
from q_logic.q_logic_logging import Advanced_stat_logger, Time_logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    """
    main DQN RL agent class
    it is used for exploration and training
    it is modular in sense that you can turn on or off a lot of standard DQN srchitecutre choices such as:
    Double DQN, N step reward, polyak update, gamma size
    
    you can also use several choices for models such as noisy net, dueling net, QR-DQN...

    to use the class you need to define a knew subclass with functions as per specifications stated in the example 
    """
    def __init__(self, model, optimizer, possible_actions ,batch_size, criterion = huberLoss(), scheduler = False, 
                 train = True, advanced_logging_path= False, time_logging_path = False, double_q=True ,
                 n_step_remember=1, gamma=0.93,memory = ReplayBuffer(), save_dir = "model_saves", polyak_update = True, noisy_net = False):
        

        self.save_dir = save_dir
        self.possible_actions = possible_actions #a dictionary of possible actions for different problems
        self.num_actions = len(possible_actions)


        self.noisy_net = noisy_net
        self.epsilon = 0.9         
        self.epsilon_min =  0   
        self.epsilon_decay =  0.995

        self.n_games = 0
        self.gamma = gamma
        self.action_counter= 0
        self.batch_size = batch_size

        #memory
        self.memory = memory 
        self.rewards_average = 0
        self.n_step_remember = n_step_remember
        self.last_action = None
        self.rewards = deque(maxlen=n_step_remember)
        self.remember_data = deque(maxlen=n_step_remember)

        #modeli
        self.is_training = train
        self.model = model.to(DEVICE)
        self.model_target =  copy.deepcopy(model).to(DEVICE)
        self.model_target.load_state_dict(self.model.state_dict())
        self.model_target.eval() 
        if noisy_net:
            self.model_target.train()


        #q-trainer
        self.trainer = QTrainer(self.model, self.model_target, scheduler=scheduler,  noisy_net= noisy_net,
                                double_q=double_q, criterion= criterion, optimizer=optimizer,gamma=gamma, polyak_update=polyak_update)
         
        # logging
        self.advanced_logger = Advanced_stat_logger(advanced_logging_path, 1000, self.batch_size) if advanced_logging_path else None
        self.time_logger =  Time_logger(time_logging_path) if time_logging_path else None


    def save_agent_state(self, file_name='agent_state.pth', training = False):
        os.makedirs(self.save_dir, exist_ok=True)
        file_path = f"{self.save_dir}/{file_name}.pt" 

        if training:
            checkpoint  ={
                "model_state_dict": self.model.state_dict(),
                "target_state_dict": self.model_target.state_dict(),
                "epsilon": self.epsilon,
                "action_counter": self.action_counter,
                "n_games": self.n_games,
                "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                "learning_rate": self.trainer.scheduler.get_lr()
            }
        else:
            checkpoint  ={
                "model_state_dict": self.model.state_dict(),
            }

        torch.save(checkpoint, file_path)
        print("agent state saved")
        

    def load_agent_state(self, file_path='agent_state.pth', training=False, noisynet = False):
        data = torch.load(file_path, map_location="cpu")

        self.model.load_state_dict(data["model_state_dict"])
        self.is_training = training
        if noisynet:
            self.model.is_training = training
        if False and training: #bakcata cemo se kasnije s ovime 
            self.model_target.load_state_dict(data["target_state_dict"])
            self.epsilon= data["epsilon"]
            self.n_games= data["n_games"]
            self.action_counter= data["action_counter"]
            self.trainer.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.learning_rate(data["learning_rate"])


    def return_counter(self):
        return self.counter

    def remember(self, data, data_novi): # saving experience to memory

        memory_state = self.get_memory_state(data) #saving data in saving friendly format
        self.remember_data.append((memory_state,self.last_action)) # saving states, used for N step remember

        reward,done = self.give_reward(data_novi = data_novi,data = data, akcija = self.last_action)
        self.rewards_average +=  self.gamma ** len(self.rewards) * reward 
        self.rewards.append(reward) # saving rewards, used for N step remember

        next_memory_state = self.get_memory_state(data_novi)

        num_visits = []
        td_error_means = []

        if done: # if the end of episode is reached
            self.n_games += 1

            gamma_train = self.gamma ** (len(self.rewards))
            while not len(self.remember_data) == 0: # going through all saved memories, all of them will have no bootstrapped reward added
                memory_state,action = self.remember_data.popleft()
                experience = (memory_state, np.argmax(action), self.rewards_average, next_memory_state, done, gamma_train)
                experience_visits, experience_td_error = self.memory.push(experience)

                if experience_visits is not None: #log handling, to see how many times ieach memory was visited
                    num_visits.append(experience_visits)
                    td_error_means.append(experience_td_error)

                gamma_train = gamma_train / self.gamma
                reward = self.rewards.popleft()
                self.rewards_average = (self.rewards_average - reward) / self.gamma
            self.rewards_average = 0
            

        elif len(self.rewards) == self.n_step_remember: # saving memories with next state for boostrapping
            memory_state, action = self.remember_data.popleft()
            experience = (memory_state, np.argmax(action), self.rewards_average, next_memory_state, done, self.gamma**self.n_step_remember)
            experience_visits, experience_td_error  = self.memory.push(experience)

            if experience_visits is not None: #log handling
                num_visits.append(experience_visits)    
                td_error_means.append(experience_td_error)        
            
            reward = self.rewards.popleft()
            self.rewards_average = (self.rewards_average - reward) / self.gamma

        if (self.advanced_logger is not None) and len(num_visits) >0:
            self.advanced_logger.remember_log(num_visits, td_error_means,self.n_games)
            

    def train(self):
        
        if not len(self.memory) <1000 :
            a = time.time()
            mini_sample, idxs, weights, sample_priorities, log_sample = self.memory.sample(self.batch_size) #sampling from memory
            vrijeme_sample = time.time()-a
            

            # Unpack the samples into separate lists
            memory_states, actions, rewards, next_memory_states, dones,gamma_train = zip(*mini_sample)

            
            game_states = self.stack_state_batch(self.memory_to_model(memory_states)) #from memory structure to train structure
            actions = torch.tensor(np.array(actions), dtype=torch.long) 
            rewards = torch.tensor(np.array(rewards), dtype=torch.float)
            next_game_states = self.stack_state_batch(self.memory_to_model(next_memory_states))
            dones = torch.tensor(np.array(dones), dtype=torch.bool) 
            gamma_train = torch.tensor(np.array(gamma_train), dtype=torch.float)
            weights = torch.tensor(np.array(weights), dtype=torch.float)

            loss, td, Q_val =self.trainer.train_step(game_states, actions, rewards, next_game_states, dones,gamma_train, weights) # giving all the data to the trainer to handle training

            a = time.time()
            self.memory.update_priorities(idxs, td, sample_priorities) # updating priorities if priority sampler is used
            vrijeme_update_priorities = time.time()-a


            # advanced logging
            a = time.time()
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)   # L2 gradient norm
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5  
            
            log_train = (td, Q_val, self.episode_count(game_states), total_norm, loss)
            vrijeme_logging = time.time()-a

            vremena_long_term = (vrijeme_sample,vrijeme_update_priorities,vrijeme_logging)

            lr = self.trainer.optimizer.param_groups[0]["lr"]

            if self.advanced_logger is not None:
                self.advanced_logger(log_train, log_sample,self.n_games,lr )
            # end of logging

            return loss # returning loss to user for quick sainity checks
        return 0

    def episode_count(self, metadata_states):
        return np.zeros((len(metadata_states),))
    
    def get_action(self, data): # this is control function for agent to interact with environment
        self.action_counter += 1
        game_state = self.get_state(data)
        game_state = self.to_device(game_state)

        
        final_move = np.zeros((self.num_actions,)) #

        if self.noisy_net and self.is_training: # if using noisy net the exploration is done by the net itself
            self.model.train() # noisy nets need to be set to train
            self.model.reset_noise() # always reset noise for noisy nets
            with torch.no_grad():
                self.model.ratios = True
                prediction, ratios = self.model(**game_state)
                prediction = prediction.cpu()
                self.model.ratios = False
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            move_direction = self.possible_actions[move]
            self.last_action = final_move
            return move_direction ,ratios 
        
        else:
            self.epsilon = max(self.epsilon_min , self.epsilon * self.epsilon_decay)
            if  random.uniform(0, 1) < self.epsilon and self.is_training: # exploration 
                move = random.randint(0, self.num_actions-1)
                final_move[move] = 1
                move_direction = self.possible_actions[move]
            else: # explotation
                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(**game_state).cpu()
                self.model.train()
                move = torch.argmax(prediction).item()
                final_move[move] = 1
                move_direction = self.possible_actions[move]
            
            

        self.last_action = final_move
        return move_direction
    
    def to_device(self, state_dict): # moving all the states to cuda
        return {k: v.to(DEVICE) for k, v in state_dict.items()}
    
    def stack_state_batch(self, tuple_dict): # stacking torch tensors for training 
        keys = tuple_dict[0].keys()
        return {k: torch.stack([s[k] for s in tuple_dict], dim=0) for k in keys}



class QTrainer:
    """
    used to handle all the TD training
    """
    def __init__(self, model,model_target,double_q=True, noisy_net = False, criterion = nn.MSELoss(), optimizer = None, scheduler = False, gamma=0.93, polyak_update = True):

        self.model = model
        self.model_target = model_target

        self.polyak_update = polyak_update
        self.polyak_tau = 0.005 
        
        self.optimizer = optimizer

        self.criterion = criterion
        self.scheduler = None
        if scheduler:
            self.scheduler = scheduler
        self.model_target_counter = 0
        self.model_target_cycle = 1000 # ovo je jako bitno 
        self.model_target_update_counter = 0

        self.double_q = double_q
        self.noisy_net = noisy_net

    def train_step(self, game_state, action, reward, next_game_state, done, gamma_train,weights):
        self.model_target_update() #updating model traget wether it is polyak or regular step update

        game_state = self.to_device(game_state) # putting all the data to cuda
        action = action.long().to(DEVICE)
        reward = reward.float().to(DEVICE)
        next_game_state = self.to_device(next_game_state)
        done = done.bool().to(DEVICE) 
        gamma_train = gamma_train.float().to(DEVICE)
        weights = weights.float().to(DEVICE)


        self.model.train()
        
        if self.noisy_net: # always reseting noise before inference for noisynet
            self.model.reset_noise()
            self.model_target.reset_noise()

        pred = self.model(**game_state)  # value prediction of the state
        
        action_indices = action.view(-1)
        pred_a = pred.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        max_next_q = self._compute_bootstrap_q(next_game_state) # value prediction for the next state
        target_a = reward + gamma_train * max_next_q * (~done) # bootrstrapped value prediciton for state



        
        loss = F.mse_loss(pred_a, target_a) # calculate td loss and back propafate
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        

        loss = loss.cpu()

        target_a = target_a.detach().cpu().numpy().squeeze()
        pred_a = pred_a.detach().cpu().numpy().squeeze()
        td_errors = np.abs(target_a - pred_a)


        return float(loss), td_errors, np.abs(pred_a)  #return all the necessary data for log and update


    def model_target_update(self):
        if self.polyak_update: # polyak continous model target update
            for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_( (1 - self.polyak_tau) * target_param.data + self.polyak_tau * param.data )

        self.model_target_counter += 1 #step target every  self.model_target_cycle moves
        if(self.model_target_counter > self.model_target_cycle  ):
            self.model_target_counter = 0
            self.model_target.load_state_dict(self.model.state_dict())
            self.model_target_update_counter += 1

    def _compute_bootstrap_q(self, next_game_state):# used for bootstrapping
        with torch.no_grad():
            q_next_target = self.model_target(**next_game_state)

        if self.double_q: # if double q net, model that is beeing trained is used to select the best next move for the target net
            with torch.no_grad():
                    q_next_online = self.model(**next_game_state)
                    next_actions = q_next_online.argmax(dim=1)
                    max_next_q = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1) # target net is used to calculate the value of the move
        else:
            # standard DQN target net is used to select the move and deterimne the value
            max_next_q, _ = q_next_target.max(dim=1)

        return max_next_q

    def to_device(self, state_dict):
        return {k: v.to(DEVICE) for k, v in state_dict.items()}




def set_seed(seed: int): #setting all seeds used in training 
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
