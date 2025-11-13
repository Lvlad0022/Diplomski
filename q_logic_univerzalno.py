
import json
import sys
import random
import time
import traceback
import copy


import torch
import random
import numpy as np
from collections import deque
# Assuming these imports are from your game and helper files
# from game import SnakeGameAI, Direction, Point
# from model import Linear_QNet, QTrainer # We will replace Linear_QNet
# from helper import plot
import json

# --- Define the AdvancedSnakeNN Model (as provided previously) ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR 
import os
from queue import Queue
from collections import deque


from snake_models import AdvancedSnakeNN, ResnetSnakeNN
from loss_functions import huberLoss
from q_logic_memory_classes import TDPriorityReplayBuffer, ReplayBuffer, RewardPriorityReplayBuffer
from q_logic_logging import Advanced_stat_logger, Time_logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QTrainer:
    """
    Trains the Q-network using experiences.
    Implements the training step for a DQN-like agent.
    """
    def __init__(self, model,model_target,double_q=True, criterion = nn.MSELoss(), optimizer = None, scheduler = False, lr=0.0005, gamma=0.93):
        """
        Initializes the QTrainer.

        Args:
            model (nn.Module): The Q-network model (e.g., AdvancedSnakeNN).
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
        """
        self.lr = lr
        self.model = model
        self.model_target = model_target
        
        self.optimizer = optimizer

        self.criterion = criterion
        self.scheduler = None
        if scheduler:
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.9999) 
        self.model_target_counter = 0
        self.model_target_cycle = 200 # ovo je jako bitno 
        self.model_target_cycle_mult = 1.2
        self.model_target_max = 3000
        self.model_target_update_counter = 0

        self.double_q = double_q

    def train_step(self, map_state, metadata_state, action, reward, next_map_state, next_metadata_state, done, gamma_train,weights):
        self.model_target_update()
        # Ensure inputs are tensors and have the correct data types
        a = time.time()
        map_state = map_state.float().to(DEVICE)
        metadata_state = metadata_state.float().to(DEVICE)
        action = action.long().to(DEVICE)
        reward = reward.float().to(DEVICE)
        next_map_state = next_map_state.float().to(DEVICE)
        next_metadata_state = next_metadata_state.float().to(DEVICE)
        done = done.bool().to(DEVICE) 
        gamma_train = gamma_train.float().to(DEVICE)
        weights = weights.float().to(DEVICE)
        vrijeme_move_to_gpu = time.time()-a


        # If only a single experience is provided, add a batch dimension
        # Check the shape of map_state to determine if it's a single experience or a batch
        if len(map_state.shape) == 3: # Single experience: (C, H, W)
             map_state = torch.unsqueeze(map_state, 0) # -> (1, C, H, W)
             metadata_state = torch.unsqueeze(metadata_state, 0) # -> (1, metadata_dim)
             action = torch.unsqueeze(action, 0) # -> (1, num_actions)
             reward = torch.unsqueeze(reward, 0) # -> (1,)
             next_map_state = torch.unsqueeze(next_map_state, 0) # -> (1, C, H, W)
             next_metadata_state = torch.unsqueeze(next_metadata_state, 0) # -> (1, metadata_dim)
             done = torch.unsqueeze(done, 0) # -> (1,)
        # Else: Assume it's already a batch: (B, C, H, W), (B, metadata_dim), etc.


         #  ----  bootstraping ------

        a = time.time() 
        self.model.train()
        pred = self.model(map_state, metadata_state) 
        
        max_next_q = self._compute_bootstrap_q(next_map_state, next_metadata_state)
        vrijeme_forward_prop = time.time()-a


        target = pred.detach().clone()
        action_indices = torch.argmax(action, dim=1) 
        target[range(len(done)), action_indices] = reward + gamma_train * max_next_q * (~done) # Use ~done for not done

        # 3: Compute the loss
        self.optimizer.zero_grad(set_to_none=True)

        pred_a = pred.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        target_a = target.gather(1, action_indices.unsqueeze(1)).squeeze(1)


        loss = self.criterion(target_a, pred_a, weights)   
        
        

        a = time.time()
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        vrijeme_back_prop = time.time()-a

        loss = loss.cpu()

        target_a = target_a.detach().cpu().numpy().squeeze()
        pred_a = pred_a.detach().cpu().numpy().squeeze()
        td_errors = np.abs(target_a - pred_a)



        vremena = (vrijeme_move_to_gpu, vrijeme_forward_prop, vrijeme_back_prop)
        return float(loss), td_errors, np.abs(pred_a), vremena


    def model_target_update(self):
        self.model_target_counter += 1
        if(self.model_target_counter > self.model_target_cycle  ):
            self.model_target_cycle = min( self.model_target_cycle_mult* self.model_target_cycle, self.model_target_max)
            self.model_target_counter = 0
            self.model_target.load_state_dict(self.model.state_dict())
            self.model_target_update_counter += 1

    def _compute_bootstrap_q(self, next_map_state, next_metadata_state):
        """
        Returns max_next_q for each sample, either DQN-style or Double DQN-style.
        """
        # Q_target(s', :)
        self.model_target.eval()
        with torch.no_grad():
            q_next_target = self.model_target(next_map_state, next_metadata_state)

        if self.double_q:
            # 🔹 Double DQN: argmax from online net, value from target net
            q_next_online = self.model(next_map_state, next_metadata_state)

            next_actions = q_next_online.argmax(dim=1)          # [B]
            max_next_q = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            # 🔹 Standard DQN
            max_next_q, _ = q_next_target.max(dim=1)

        return max_next_q



# --- Constants ---
MAX_MEMORY = 100_000
BATCH_SIZE = 32
LR = 0.0005

# --- Agent Class ---
class Agent:

    def __init__(self, model, optimizer ,criterion = huberLoss() ,train = True, advanced_logging_path= False, time_logging_path = False, double_q=True ,
                 n_step_remember=1, gamma=0.93,memory = ReplayBuffer(), batch_size = BATCH_SIZE):
        
        self.n_games = 0
        self.epsilon = 0.9           
        self.epsilon_min = 0.05      
        self.epsilon_decay = 0.9995 
        self.gamma = gamma
        self.action_counter= 0
        self.train = train
        self.batch_size = batch_size

        #memory
        self.memory = memory 
        self.rewards_average = 0
        self.n_step_remember = n_step_remember
        self.last_action = None
        self.rewards = deque(maxlen=n_step_remember)
        self.remember_data = deque(maxlen=n_step_remember)

        #modeli
        self.model = model.to(DEVICE)
        self.model_target =  copy.deepcopy(model).to(DEVICE)
        self.model_target.load_state_dict(self.model.state_dict())
        self.model_target.eval() 

        self.trainer = QTrainer(self.model, self.model_target, double_q=double_q, criterion= criterion, optimizer=optimizer, lr = LR,gamma=gamma)
         
        #logging
        self.advanced_logger = Advanced_stat_logger(advanced_logging_path, 1000, self.batch_size) if advanced_logging_path else None
        self.time_logger =  Time_logger(time_logging_path) if time_logging_path else None


    def return_counter(self):
        return self.counter

    def save_agent_state(self, file_path='agent_state.pth'):
        """Sprema stanje agenta i memoriju u odvojene datoteke."""
        

    def load_agent_state(self, file_name='agent_state.pth', training=True):
        """Učitava stanje agenta i memoriju iz odvojenih datoteka."""
        

    def change_weights(self, other):
        self.model.load_state_dict(other.model.state_dict())

    def get_model_state(self):
        return self.n_games, self.epsilon, self.trainer.optimizer.param_groups[0]["lr"]


    def remember(self, data, data_novi):
        map_state, metadata_state = self.get_state(data) 
        self.remember_data.append((map_state, metadata_state,self.last_action))

        reward,done = self.give_reward(data_novi = data_novi,data = data, akcija = self.last_action)
        self.rewards_average +=  self.gamma ** len(self.rewards) * reward #len(self.rewards) = len(remember_data)-1
        self.rewards.append(reward)

        next_map_state, next_metadata_state = self.get_state(data_novi)
        
        if done:
            gamma_train = self.gamma ** (len(self.rewards))
            while not len(self.remember_data) == 0:
                map_state, metadata_state,action = self.remember_data.popleft()
                experience = (map_state, metadata_state, action, self.rewards_average, next_map_state, next_metadata_state, done, gamma_train)
                self.memory.push(experience)
                gamma_train = gamma_train / self.gamma
                reward = self.rewards.popleft()
                self.rewards_average = (self.rewards_average - reward) / self.gamma
            self.rewards_average = 0
            

        elif len(self.rewards) == self.n_step_remember:
            map_state, metadata_state, action = self.remember_data.popleft()
            experience = (map_state, metadata_state, action, self.rewards_average, next_map_state, next_metadata_state, done, self.gamma**self.n_step_remember)
            self.memory.push(experience)
            reward = self.rewards.popleft()
            self.rewards_average = (self.rewards_average - reward) / self.gamma
            

    def train_long_term(self):
        """Trains the model using a batch of experiences from the memory."""
        if self.n_games > 500 and isinstance(self.model, ResnetSnakeNN):
            self.model.freeze_backbone(False)

        
        if not len(self.memory) <1000 :
            a = time.time()
            mini_sample, idxs, weights, sample_priorities, log_sample = self.memory.sample(self.batch_size)
            vrijeme_sample = time.time()-a
            

            # Unpack the samples into separate lists
            map_states, metadata_states, actions, rewards, next_map_states, next_metadata_states, dones,gamma_train = zip(*mini_sample)
            # Convert lists of numpy arrays to tensors
            map_states = torch.tensor(np.array(map_states), dtype=torch.float)
            metadata_states = torch.tensor(np.array(metadata_states), dtype=torch.float)
            actions = torch.tensor(np.array(actions), dtype=torch.long) # Actions are typically long
            rewards = torch.tensor(np.array(rewards), dtype=torch.float)
            next_map_states = torch.tensor(np.array(next_map_states), dtype=torch.float)
            next_metadata_states = torch.tensor(np.array(next_metadata_states), dtype=torch.float)
            dones = torch.tensor(np.array(dones), dtype=torch.bool) # Done flags are boolean
            gamma_train = torch.tensor(np.array(gamma_train), dtype=torch.float)
            weights = torch.tensor(np.array(weights), dtype=torch.float)

            # --- Call the trainer with separate inputs ---
            # You need to modify your QTrainer.train_step to accept these
            loss, td, Q_val, vremena_train =self.trainer.train_step(map_states, metadata_states, actions, rewards, next_map_states, next_metadata_states, dones,gamma_train, weights)
            #print("Training long memory batch...") # Placeholder

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
            total_norm = total_norm ** 0.5  #
            
            log_train = (td, Q_val, self.episode_count(metadata_states), total_norm, loss)
            vrijeme_logging = time.time()-a

            vremena_long_term = (vrijeme_sample,vrijeme_update_priorities,vrijeme_logging)
        
            if self.time_logger is not None:
                self.time_logger(vremena_long_term, vremena_train,self.n_games)
            if self.advanced_logger is not None:
                self.advanced_logger(log_train, log_sample,self.n_games)

            return loss
        return 0

    def episode_count(self, metadata_states):
        return np.zeros((len(metadata_states),))
    
    def get_action(self, data):
        self.action_counter += 1
        """
        Chooses an action based on the current state using an epsilon-greedy strategy.

        Args:
            map_state (np.ndarray): The current map state.
            metadata_state (np.ndarray): The current metadata state.

        Returns:
            list: The chosen action (one-hot encoded).
        """
        # random moves: tradeoff exploration / exploitation
        # Decay epsilon as games progress
        map_state, metadata_state = self.get_state(data)
        
        
        final_move = [0,0,0,0] # Assuming 4 possible actions (straight, right, left)

        
        if(self.action_counter%10==0 and self.epsilon):
            self.epsilon = max(self.epsilon_min , self.epsilon * self.epsilon_decay)
        if random.uniform(0, 1) < self.epsilon: # Increased random range for slower decay
            # Exploration: Choose a random action
            move = random.randint(0, 3)
            final_move[move] = 1
            move_direction = VALID_DIRECTIONS[move]
        else:
            # Exploitation: Get prediction from the model
            # Convert numpy arrays to tensors and add batch dimension (size 1)
            map_state_tensor = torch.tensor(map_state[np.newaxis, ...], dtype=torch.float).to(DEVICE)
            metadata_state_tensor = torch.tensor(metadata_state[np.newaxis, ...], dtype=torch.float).to(DEVICE)

            # Get prediction from the model
            
            
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(map_state_tensor, metadata_state_tensor).cpu()
            self.model.train()            # Set model back to training mode after inference
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            move_direction = VALID_DIRECTIONS[move]

        self.last_action = final_move
        return move_direction


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

