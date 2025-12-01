
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

# --- Define the AdvancedSnakeNN Model (as provided previously) ---

from collections import deque
from q_logic.loss_functions import huberLoss
from q_logic.q_logic_memory_classes import TDPriorityReplayBuffer, ReplayBuffer, RewardPriorityReplayBuffer
from q_logic.q_logic_logging import Advanced_stat_logger, Time_logger
from q_logic.q_logic_schedulers import LossAdaptiveLRScheduler, TDAdaptiveScheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QTrainer:
    """
    Trains the Q-network using experiences.
    Implements the training step for a DQN-like agent.
    """
    def __init__(self, model,model_target,double_q=True, noisy_net = False, criterion = nn.MSELoss(), optimizer = None, scheduler = False, gamma=0.93, polyak_update = True):
        """
        Initializes the QTrainer.

        Args:
            model (nn.Module): The Q-network model (e.g., AdvancedSnakeNN).
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
        """
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
        self.model_target_update()

        game_state = self.to_device(game_state)
        action = action.long().to(DEVICE)
        reward = reward.float().to(DEVICE)
        next_game_state = self.to_device(next_game_state)
        done = done.bool().to(DEVICE) 
        gamma_train = gamma_train.float().to(DEVICE)
        weights = weights.float().to(DEVICE)


        self.model.train()
        
        if self.noisy_net:
            self.model.reset_noise()
            self.model_target.reset_noise()
        pred = self.model(**game_state) 
        
        action_indices = action.view(-1)
        pred_a = pred.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        max_next_q = self._compute_bootstrap_q(next_game_state)
        target_a = reward + gamma_train * max_next_q * (~done)


        # 3: Compute the loss

        
        loss = F.mse_loss(pred_a, target_a)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        

        loss = loss.cpu()

        target_a = target_a.detach().cpu().numpy().squeeze()
        pred_a = pred_a.detach().cpu().numpy().squeeze()
        td_errors = np.abs(target_a - pred_a)

        return float(loss), td_errors, np.abs(pred_a)


    def model_target_update(self):
        if self.polyak_update:
            for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_( (1 - self.polyak_tau) * target_param.data + self.polyak_tau * param.data )

        self.model_target_counter += 1
        if(self.model_target_counter > self.model_target_cycle  ):
            self.model_target_counter = 0
            self.model_target.load_state_dict(self.model.state_dict())
            self.model_target_update_counter += 1

    def _compute_bootstrap_q(self, next_game_state):
        """
        Returns max_next_q for each sample, either DQN-style or Double DQN-style.
        """
        # Q_target(s', :)
        with torch.no_grad():
            q_next_target = self.model_target(**next_game_state)

        if self.double_q:
            # 🔹 Double DQN: argmax from online net, value from target net
            with torch.no_grad():
                    q_next_online = self.model(**next_game_state)
                    next_actions = q_next_online.argmax(dim=1)
                    max_next_q = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            # 🔹 Standard DQN
            max_next_q, _ = q_next_target.max(dim=1)

        return max_next_q

    def to_device(self, state_dict):
        return {k: v.to(DEVICE) for k, v in state_dict.items()}



class Agent:

    def __init__(self, model, optimizer, possible_actions ,batch_size, criterion = huberLoss(), scheduler = False, 
                 train = True, advanced_logging_path= False, time_logging_path = False, double_q=True ,
                 n_step_remember=1, gamma=0.93,memory = ReplayBuffer(), save_dir = "model_saves", polyak_update = True, noisy_net = False):
        

        self.save_dir = save_dir
        self.possible_actions = possible_actions
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

        self.trainer = QTrainer(self.model, self.model_target, scheduler=scheduler,  noisy_net= noisy_net,
                                double_q=double_q, criterion= criterion, optimizer=optimizer,gamma=gamma, polyak_update=polyak_update)
         
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
        if training:
            self.model_target.load_state_dict(data["target_state_dict"])
            self.epsilon= data["epsilon"]
            self.n_games= data["n_games"]
            self.action_counter= data["action_counter"]
            self.trainer.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.learning_rate(data["learning_rate"])


    def return_counter(self):
        return self.counter

    def remember(self, data, data_novi):
        memory_state = self.get_memory_state(data)
        self.remember_data.append((memory_state,self.last_action))

        reward,done = self.give_reward(data_novi = data_novi,data = data, akcija = self.last_action)
        self.rewards_average +=  self.gamma ** len(self.rewards) * reward #len(self.rewards) = len(remember_data)-1
        self.rewards.append(reward)

        next_memory_state = self.get_memory_state(data_novi)

        num_visits = []
        td_error_means = []

        if done:
            self.n_games += 1

            gamma_train = self.gamma ** (len(self.rewards))
            while not len(self.remember_data) == 0:
                memory_state,action = self.remember_data.popleft()
                experience = (memory_state, np.argmax(action), self.rewards_average, next_memory_state, done, gamma_train)
                experience_visits, experience_td_error = self.memory.push(experience)
                if experience_visits is not None: #log handling
                    num_visits.append(experience_visits)
                    td_error_means.append(experience_td_error)

                gamma_train = gamma_train / self.gamma
                reward = self.rewards.popleft()
                self.rewards_average = (self.rewards_average - reward) / self.gamma
            self.rewards_average = 0
            

        elif len(self.rewards) == self.n_step_remember:
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
        """Trains the model using a batch of experiences from the memory."""

        
        if not len(self.memory) <1000 :
            a = time.time()
            mini_sample, idxs, weights, sample_priorities, log_sample = self.memory.sample(self.batch_size)
            vrijeme_sample = time.time()-a
            

            # Unpack the samples into separate lists
            memory_states, actions, rewards, next_memory_states, dones,gamma_train = zip(*mini_sample)

            
            game_states = self.stack_state_batch(self.memory_to_model(memory_states))
            actions = torch.tensor(np.array(actions), dtype=torch.long) 
            rewards = torch.tensor(np.array(rewards), dtype=torch.float)
            next_game_states = self.stack_state_batch(self.memory_to_model(next_memory_states))
            dones = torch.tensor(np.array(dones), dtype=torch.bool) # Done flags are boolean
            gamma_train = torch.tensor(np.array(gamma_train), dtype=torch.float)
            weights = torch.tensor(np.array(weights), dtype=torch.float)

            loss, td, Q_val =self.trainer.train_step(game_states, actions, rewards, next_game_states, dones,gamma_train, weights)

            a = time.time()
            self.memory.update_priorities(idxs, td, sample_priorities)
            vrijeme_update_priorities = time.time()-a


            #provjeri koliko vremena treba
            a = time.time()
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)   # L2 norma gradijenta parametra
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5  
            
            log_train = (td, Q_val, self.episode_count(game_states), total_norm, loss)
            vrijeme_logging = time.time()-a

            vremena_long_term = (vrijeme_sample,vrijeme_update_priorities,vrijeme_logging)

            lr = self.trainer.optimizer.param_groups[0]["lr"]

            if self.advanced_logger is not None:
                self.advanced_logger(log_train, log_sample,self.n_games,lr )

            return loss
        return 0

    def episode_count(self, metadata_states):
        return np.zeros((len(metadata_states),))
    
    def get_action(self, data):
        self.action_counter += 1
        game_state = self.get_state(data)
        game_state = self.to_device(game_state)

        
        final_move = np.zeros((self.num_actions,)) 

        if self.noisy_net and self.is_training:
            self.model.train() 
            self.model.reset_noise()
            with torch.no_grad():
                self.model.ratios = True
                prediction, ratios = self.model(**game_state)
                prediction = prediction.cpu()
                self.model.ratios = False
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            move_direction = self.possible_actions[move]
            self.last_action = final_move
            return move_direction, ratios  
        
        else:
            self.epsilon = max(self.epsilon_min , self.epsilon * self.epsilon_decay)
            if  random.uniform(0, 1) < self.epsilon and self.is_training: # Increased random range for slower decay
                # Exploration: Choose a random action
                move = random.randint(0, self.num_actions-1)
                final_move[move] = 1
                move_direction = self.possible_actions[move]
            else:
                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(**game_state).cpu()
                self.model.train()
                move = torch.argmax(prediction).item()
                final_move[move] = 1
                move_direction = self.possible_actions[move]
            
            

        self.last_action = final_move
        return move_direction,
    
    def to_device(self, state_dict):
        return {k: v.to(DEVICE) for k, v in state_dict.items()}
    
    def stack_state_batch(self, tuple_dict):
        """
        Given a tuple of dicts with identical keys and tensor values,
        return a dict with the same keys and stacked tensors along dim 0.
        """
        keys = tuple_dict[0].keys()
        return {k: torch.stack([s[k] for s in tuple_dict], dim=0) for k in keys}



# Default values
DEFAULT_AGENT_ID = "k"
DEFAULT_NAME="Chat GPDiddy" 
VALID_DIRECTIONS = ["up", "right", "down", "left"]
MODEL_PATH = ""
VALID_MODES = ["up", "down", "left", "right", "random", "timeout", "apple", "survive"]
BASE_DELAY = 0.05  # 50ms

number_of_moves = []
score_diff_arr = []
score_arr = []
survived_arr = []


def set_seed(seed: int):
    # Python / OS / NumPy
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch CPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic ops (optional but recommended)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
