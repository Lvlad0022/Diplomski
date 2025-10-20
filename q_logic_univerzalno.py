
import json
import sys
import random
import time
import traceback

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
from loss_functions import huberPriorityLoss, PriorityLoss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QTrainer:
    """
    Trains the Q-network using experiences.
    Implements the training step for a DQN-like agent.
    """
    def __init__(self, model,model_target, criterion = nn.MSELoss(), optimizer = None, scheduler = False, lr=0.0005, gamma=0.93):
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

    def train_step(self, map_state, metadata_state, action, reward, next_map_state, next_metadata_state, done, gamma_train,end_priority):
        self.model_target_update()
        # Ensure inputs are tensors and have the correct data types
        map_state = map_state.float().to(DEVICE)
        metadata_state = metadata_state.float().to(DEVICE)
        action = action.long().to(DEVICE)
        reward = reward.float().to(DEVICE)
        next_map_state = next_map_state.float().to(DEVICE)
        next_metadata_state = next_metadata_state.float().to(DEVICE)
        done = done.bool().to(DEVICE) 
        gamma_train = gamma_train.float().to(DEVICE)
        end_priority = end_priority.float().to(DEVICE)


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
             end_priority = torch.unsqueeze(end_priority, 0)
        # Else: Assume it's already a batch: (B, C, H, W), (B, metadata_dim), etc.


        # 1: Predicted Q values with current state
        # Set model to training mode before the forward pass for training
        
        self.model.train()
        pred = self.model(map_state, metadata_state) # Shape: (batch_size, num_actions)

        # 2: Calculate the target Q values based on the Bellman equation
        target = pred.detach().clone()
        with torch.no_grad(): # Calculate targets without tracking gradients
            # Set model to evaluation mode for calculating target Qs from next state
            self.model_target.eval()
            next_state_q_values = self.model_target(next_map_state, next_metadata_state) # Shape: (batch_size, num_actions)
            # Set model back to training mode
            self.model.train()

            max_next_q = torch.max(next_state_q_values, dim=1)[0] # Get max Q for each item in batch

        # Update the target Q-value for the action that was actually taken
        # target[idx][action_index] = reward[idx] + self.gamma * max_next_q[idx] * (1 - done[idx])
        # Using advanced indexing for efficiency instead of loop
        target[range(len(done)), torch.argmax(action, dim=1)] = reward + gamma_train * max_next_q * (~done) # Use ~done for not done

        # 3: Compute the loss
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred, end_priority)
        
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        return float(loss)

    def model_target_update(self):
        self.model_target_counter += 1
        if(self.model_target_counter > self.model_target_cycle  ):
            self.model_target_cycle = min( self.model_target_cycle_mult* self.model_target_cycle, self.model_target_max)
            self.model_target_counter = 0
            self.model_target.load_state_dict(self.model.state_dict())




class RewardPropReplayBuffer:
    def __init__(self,priority_mult = 2, capacity=100_000, gamma=0.93, n_step_remember = 1):
        self.buffer = []
        self.priorities = []
        self.predecesor = []
        self.gamma = gamma
        self.capacity = capacity
        self.counter = 0
        self.n_step = n_step_remember
        self.priority_mult = priority_mult

    def push(self, experience, reward, predecesor): #tu treba samo paziti da predecesor bude false na prvih n_step_remember stanja u novoj igri
        priority = 1 + self.priority_mult * reward  
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
            if predecesor:
                self.predecesor.append(len(self.buffer)-1 - self.n_step)
        else:
            self.buffer[self.counter] = experience
            self.priorities[self.counter] = priority
            if predecesor:
                self.predecesor[self.counter] = (self.counter - self.n_step) % self.capacity
            self.counter = (self.counter+1) % self.capacity

    def sample(self, batch_size):
        # pretvori prioritete u vjerojatnosti
        p = np.maximum(np.array(self.priorities), 0) + 1e-6
        probs = p / p.sum()
        idxs = np.random.choice(len(    ), batch_size, p=probs)
        samples = [self.buffer[i] for i in idxs]
        return samples, idxs

    def update_after_train(self, indices, rewards):
        for idx, r in zip(indices, rewards):
            # smanji trenirano stanje
            self.priorities[idx] *= self.gamma**2

            # propagiraj reward unazad ako postoji prethodnik
            if idx > 0:
                self.priorities[self.predecesor[idx]] += r * self.gamma

        # osiguraj da prioriteti ne postanu negativni ili preveliki
        self.priorities = deque(np.clip(self.priorities, 0, 10), maxlen=self.buffer.maxlen)






# --- Constants ---
MAX_MEMORY = 100_000
BATCH_SIZE = 32
LR = 0.0005

# --- Item and Move Mappings ---
# (Keep your existing mappings)


# --- Agent Class ---
class Agent:

    def __init__(self, model, optimizer ,criterion = nn.MSELoss() ,train = True,n_step_remember=1, gamma=0.93,end_priority = False):
        """
        Initializes the Agent.

        Args:
            map_channels (int): Number of channels in the input map image.
            map_height (int): Height of the input map image.
            map_width (int): Width of the input map image.
            metadata_dim (int): Dimension of the metadata input.
            num_actions (int): Number of possible actions.
        """
        self.n_games = 0
        self.epsilon = 0.9           # Početna vrijednost
        self.epsilon_min = 0.05      # Minimalna vrijednost
        self.epsilon_decay = 0.9995 
        if end_priority:
            self.memory =RewardPropReplayBuffer(priority_mult = end_priority, capacity=MAX_MEMORY, gamma= gamma, n_step_remember=n_step_remember) 
        else:
            self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.end_priority = end_priority
        self.episode_count = 0

        # --- Use the AdvancedSnakeNN model --
        self.model = model.to(DEVICE)
        self.model_target = model.to(DEVICE)
        # Obavezno sinkronizirajte težine na početku!
        self.model_target.load_state_dict(self.model.state_dict())

        self.trainer = QTrainer(self.model,self.model_target, criterion= criterion, optimizer=optimizer, lr = LR,gamma=gamma)

        
        self.counter= 0 
        
        self.n_step_remember = n_step_remember
        self.last_action = None
        self.rewards = deque(maxlen=n_step_remember)
        self.remember_data = deque(maxlen=n_step_remember)
        self.gamma = gamma
        self.rewards_average = 0
        self.train = train


    def return_counter(self):
        return self.counter

    def save_agent_state(self, file_path='agent_state.pth'):
        """Sprema stanje agenta i memoriju u odvojene datoteke."""
        
        # Putanja za glavnu datoteku sa stanjem
        
        # Putanja za odvojenu datoteku s memorijom
        memory_file_path = file_path.replace('.pth', '_memory.pth')

        # 1. Stvori rječnik sa SVIM komponentama OSIM memorije
        agent_state = {
            'n_games': self.n_games,
            'epsilon': self.epsilon,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
        }

        # 2. Spremi stanje agenta (bez memorije)
        torch.save(agent_state, file_path)
        print(f"Stanje agenta spremljeno u: {file_path}")

        # 3. Spremi memoriju u ZASEBNU datoteku
        num_experiences_to_save = 20000 
    
        # Uzmi samo zadnjih N iskustava iz deque-a
        # Prvo pretvorimo u listu da možemo koristiti slicing
        recent_memory = list(self.memory)[-num_experiences_to_save:]
        
        torch.save(recent_memory, memory_file_path)
        print(f"Spremljeno zadnjih {len(recent_memory)} iskustava u: {memory_file_path}")
        # Unutar vaše Agent klase
    
    def get_model_state(self):
        lr = LR
        if self.trainer.optimizer:
            lr = self.trainer.optimizer.param_groups[0]["lr"]
        return self.n_games, self.epsilon, lr

    def load_agent_state(self, file_name='agent_state.pth', training=True):
        """Učitava stanje agenta i memoriju iz odvojenih datoteka."""
        model_folder_path = './model'
        state_file_path = os.path.join(model_folder_path, file_name)
        memory_file_path = state_file_path.replace('.pth', '_memory.pth')

        # Provjeri postoje li obje datoteke
        if not os.path.exists(state_file_path) or not os.path.exists(memory_file_path):
            print(f"Upozorenje: Jedna od datoteka za učitavanje nije pronađena. Agent počinje od nule.")
            return False

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 1. Učitaj stanje agenta (bez memorije)
            checkpoint = torch.load(state_file_path, map_location=device)
            print(f"Učitavanje stanja agenta s putanje: {state_file_path}")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if training:
                #if 'optimizer_state_dict' in checkpoint:
                #    self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # 2. Učitaj memoriju iz ZASEBNE datoteke
                print(f"Učitavanje memorije s putanje: {memory_file_path}")
                memory_list = torch.load(memory_file_path, map_location=device)
                self.memory = deque(memory_list, maxlen=self.memory.maxlen)


                self.n_games = checkpoint.get('n_games', self.n_games)
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.model.train()
                print(f"Agent učitan u 'training' modu.")
            else:
                self.model.eval()
                self.epsilon = 0
                print("Agent učitan u 'inference' modu.")

            self.model_target.load_state_dict(self.model.state_dict())
            self.model_target.eval()
            
            print(f"Stanje uspješno učitano. Nastavlja se od {self.n_games}. igre.")
            print(f"Veličina učitane memorije je: {len(self.memory)}")
            return True

        except Exception as e:
            print(f"Greška prilikom učitavanja stanja agenta: {e}")
            traceback.print_exc()
            return False

    def change_weights(self, other):
        self.model.load_state_dict(other.model.state_dict())

    '''
    def dead_from_hitting_the_wall(self, previous_head_pos, move_direction, game_map):
        fatal_pos = previous_head_pos.copy() # Napravi kopiju da ne mijenjaš original
        if move_direction == 'up':
            fatal_pos['row'] -= 1
        elif move_direction == 'down':
            fatal_pos['row'] += 1
        elif move_direction == 'left':
            fatal_pos['column'] -= 1
        elif move_direction == 'right':
            fatal_pos['column'] += 1

        # 2. Provjeri je li pozicija izvan granica mape
        rows, cols = len(game_map), len(game_map[0])
        if not (0 <= fatal_pos['row'] < rows and 0 <= fatal_pos['column'] < cols):
            return "WALL_COLLISION_OUT_OF_BOUNDS" # Igrač je izašao izvan mape

        # 3. Provjeri što se nalazi na ćeliji sudara
        cell_content = game_map[fatal_pos['row']][fatal_pos['column']]
        
        if cell_content and cell_content.get('type') == 'border':
            return 0
            
        if cell_content and cell_content.get('type') == 'snake-body':
            return 1
            
        if cell_content and cell_content.get('type') == 'snake-head':
            return 2
            
        return 3
    '''

    def remember(self, data, data_novi):
        map_state, metadata_state = self.get_state(data) 
        self.remember_data.append((map_state, metadata_state,self.last_action))

        reward = self.give_reward(data_novi = data_novi,data = data, akcija = self.last_action)
        self.rewards_average +=  self.gamma ** len(self.rewards) * reward 
        self.rewards.append(reward)

        next_map_state, next_metadata_state = self.get_state(data_novi)
        done = 1 if data_novi.get("winner") is not None else 0

        predecesor =True
        if(self.episode_count < self.n_step_remember):
            predecesor = False
        
        if done:
            gamma_train = self.gamma ** (len(self.rewards))
            while not len(self.remember_data) == 0:
                if self.end_priority:
                    self.memory.push((map_state, metadata_state, action, self.rewards_average, next_map_state, next_metadata_state, done, gamma_train), self.rewards_average, predecesor)
                    gamma_train = gamma_train / self.gamma
                    reward = self.rewards.popleft()
                    self.rewards_average = (self.rewards_average - reward) / self.gamma
                else:
                    map_state, metadata_state,action = self.remember_data.popleft()
                    self.memory.append((map_state, metadata_state, action, self.rewards_average, next_map_state, next_metadata_state, done, gamma_train, self.end_priority))
                    gamma_train = gamma_train / self.gamma
                    reward = self.rewards.popleft()
                    self.rewards_average = (self.rewards_average - reward) / self.gamma
            self.rewards_average = 0
            self.episode_count = 0
            

        elif len(self.rewards) == self.n_step_remember:
            if self.end_priority:
                self.memory.push((map_state, metadata_state, action, self.rewards_average, next_map_state, next_metadata_state, done,self.gamma ** self.n_step_remember), self.rewards_average, predecesor)
            else:
                map_state, metadata_state, action = self.remember_data.popleft()
                self.memory.append((map_state, metadata_state, action, self.rewards_average, next_map_state, next_metadata_state, done,self.gamma ** self.n_step_remember, 1))
                reward = self.rewards.popleft()
                self.rewards_average = (self.rewards_average - reward) / self.gamma
            self.episode_count += 1

    def train_long_term(self):
        """Trains the model using a batch of experiences from the memory."""
        if self.n_games > 500 and isinstance(self.model, ResnetSnakeNN):
            self.model.freeze_backbone(False)

        
        if not len(self.memory) <1000 :
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
            

            # Unpack the samples into separate lists
            map_states, metadata_states, actions, rewards, next_map_states, next_metadata_states, dones,gamma_train, end_priority = zip(*mini_sample)

            # Convert lists of numpy arrays to tensors
            map_states = torch.tensor(np.array(map_states), dtype=torch.float)
            metadata_states = torch.tensor(np.array(metadata_states), dtype=torch.float)
            actions = torch.tensor(np.array(actions), dtype=torch.long) # Actions are typically long
            rewards = torch.tensor(np.array(rewards), dtype=torch.float)
            next_map_states = torch.tensor(np.array(next_map_states), dtype=torch.float)
            next_metadata_states = torch.tensor(np.array(next_metadata_states), dtype=torch.float)
            dones = torch.tensor(np.array(dones), dtype=torch.bool) # Done flags are boolean
            gamma_train = torch.tensor(np.array(gamma_train), dtype=torch.float)
            end_priority = torch.tensor(np.array(end_priority), dtype=torch.float)

            # --- Call the trainer with separate inputs ---
            # You need to modify your QTrainer.train_step to accept these
            loss =self.trainer.train_step(map_states, metadata_states, actions, rewards, next_map_states, next_metadata_states, dones,gamma_train, end_priority)
            #print("Training long memory batch...") # Placeholder
            return loss
        return 0


    def get_action(self, data):
        self.counter += 1
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

        
        if(self.counter%10==0 and self.epsilon):
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
            
            self.model_target.eval()
            with torch.no_grad(): # No gradient calculation during inference
                prediction = self.model_target(map_state_tensor, metadata_state_tensor).cpu()
            # Set model back to training mode after inference
            self.model_target.train()
            # Get the action with the highest predicted Q-val
            # ue
            #mjere = (torch.max(prediction), torch.min(prediction), torch.mean(predition))
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

