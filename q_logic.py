
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

from snake_models import AdvancedSnakeNN, ResnetSnakeNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QTrainer:
    """
    Trains the Q-network using experiences.
    Implements the training step for a DQN-like agent.
    """
    def __init__(self, model,model_target, scheduler = False, lr=0.0005, gamma=0.93):
        """
        Initializes the QTrainer.

        Args:
            model (nn.Module): The Q-network model (e.g., AdvancedSnakeNN).
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.model_target = model_target
        
        if isinstance(self.model, AdvancedSnakeNN):
            self.optimizer = optim.Adam(model.parameters(),lr=5e-4)  

        if isinstance(self.model, ResnetSnakeNN):
            self.optimizer = optim.Adam([
                    {'params': self.model.backbone.parameters(), 'lr': 5e-5}, 
                    {'params': self.model.metadata_fc_layers.parameters()},
                    {'params': self.model.combined_fc_layers.parameters()},
                    {'params': self.model.output_layer.parameters()}
                ], lr=5e-4)  
        self.criterion = nn.MSELoss() 
        self.scheduler = None
        if scheduler:
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.9999) 
        self.model_target_counter = 0
        self.model_target_cycle = 1000

    def train_step(self, map_state, metadata_state, action, reward, next_map_state, next_metadata_state, done):
        self.model_target_update()
        # Ensure inputs are tensors and have the correct data types
        map_state = map_state.float().to(DEVICE)
        metadata_state = metadata_state.float().to(DEVICE)
        action = action.long().to(DEVICE)
        reward = reward.float().to(DEVICE)
        next_map_state = next_map_state.float().to(DEVICE)
        next_metadata_state = next_metadata_state.float().to(DEVICE)
        done = done.bool().to(DEVICE) 

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
        target[range(len(done)), torch.argmax(action, dim=1)] = reward + self.gamma * max_next_q * (~done) # Use ~done for not done

        # 3: Compute the loss
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        return float(loss)

    def model_target_update(self):
        self.model_target_counter += 1
        if(self.model_target_counter % self.model_target_cycle  == 0):
            self.model_target_counter = 0
            self.model_target.load_state_dict(self.model.state_dict())








# --- Constants ---
MAX_MEMORY = 100_000
BATCH_SIZE = 32
LR = 0.0005

# --- Item and Move Mappings ---
# (Keep your existing mappings)
items_names = {"apple": 1,
           "golden-apple": 2,
           "katana":3,
           "armour":4,
           "shorten":5,
           "tron":6,
           "freeze":7,
           "leap":8,
           "nausea":9,
           "reset-borders":10}
items_names_to_players = {
    "katana":6,
    "armour":8,
    "tron":10,
    "freeze":12,
    "loop":14, # Note: 'loop' was in your code, but 'tron' and 'freeze' were also in lasting_items and items_names_to_players. Double check these mappings.
    "golden-apple":16
}
lasting_items = [
    "katana",
    "armour",
    "tron",
    "freeze",
    "loop", # Check if 'loop' is a valid item type
    "golden-apple"]
move_names = {"up": 0, # Changed to 0, 1, 2 to match typical 3-action output (straight, right, left)
              "right": 1,
              "down": 2, # Assuming 'down' is same action as 'up' in terms of turning (straight)
              "left": 3} # Assuming 'left' is a turn left

# --- Agent Class ---
class Agent:

    def __init__(self, player1_name):
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
        self.epsilon_min = 0.05        # Minimalna vrijednost
        self.epsilon_decay = 0.99995 
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()

        # --- Use the AdvancedSnakeNN model --
        self.model = AdvancedSnakeNN().to(DEVICE)
        self.model_target = AdvancedSnakeNN().to(DEVICE)
        # Obavezno sinkronizirajte težine na početku!
        self.model_target.load_state_dict(self.model.state_dict())

        self.trainer = QTrainer(self.model,self.model_target, lr = LR)

        self.player1_name = player1_name
        self.player2_name = None
        self.player1_id = None # Store player IDs
        self.player2_id = None
        self.enemy_cons = 1 # Placeholder, adjust based on game state if available
        self.counter= 0 
        self.last_action = None

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
        if self.trainer.scheduler:
            lr = self.trainer.scheduler.get_last_lr()[0]
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

    # Unutar Agent klase u q_logic.py
    def give_reward(self, data, data_prosli):
        winner = data.get("winner")
        if winner is not None:
            self.n_games += 1
            return 1.0 if winner == self.player1_name else -1.0

        # living bonus
        reward = 0.01

        # immediate scoring (difference)
        p1_now  = data["players"][self.player1_id]["score"]
        p1_prev = data_prosli["players"][self.player1_id]["score"]
        p2_now  = data["players"][self.player2_id]["score"]
        p2_prev = data_prosli["players"][self.player2_id]["score"]

        score_gain = (p1_now - p1_prev) /10
        reward += 0.001 * score_gain   # <- easy to feel; tune 0.005–0.02

        # (optional) late-game pressure if behind
        move_count = data.get("moveCount", 0)
        if move_count > 700:
            lead = (p1_now - p2_now)
            reward += 0.001 * np.tanh(lead / 5.0)

        return reward


    def get_state(self, data):
        """
        Processes the game state JSON data into map and metadata arrays.

        Args:
            data (dict): The game state data loaded from the JSON file.

        Returns:
            tuple: (metadata_array, map_array)
        """
        map_height = 25
        map_width = 60
        map_json = data["map"]
        players_json = data["players"]
        game_info = data 


        if(self.player2_name == None):# saving ids and names into agents memory if not already done 
            if(self.player1_name == players_json[0]["name"]):
                self.player1_id = 0
                self.player2_id = 1
                self.player2_name = players_json[1]["name"]
            else:
                self.player1_name = players_json[1]["name"]
                self.player1_id = 1
                self.player2_id = 0
                self.player2_name = players_json[0]["name"]
        
        map_array = np.zeros((map_height, map_width, 3), dtype=np.float32) # Use float32 for torch


        # Populate the map array
        for i, row in enumerate(map_json):
            for j, item in enumerate(row):
                if item is not None:
                    item_type = item.get("type") # Use .get() for safety
                    if item_type == "border":
                        map_array[i, j, 0] = 1  # Border channel
                    elif item_type in ["snake-body", "snake-head"]:
                        player_name = item.get("playerName")
                        if player_name == self.player1_name:
                            # Your snake body/head
                            map_array[i, j, 1] = 1 if item_type == "snake-head" else 0.5 # Your snake channel
                        elif player_name == self.player2_name:
                            # Enemy snake body/head
                            map_array[i, j, 1] = -1 if item_type == "snake-head" else -0.5 # Enemy snake channel
                    elif item_type in items_names:
                        # Item channel - use normalized item ID
                        map_array[i, j, 2] = items_names[item_type] / len(items_names) # Normalize item ID

        # --- Metadata Processing ---
        # Create arrays for player 1 and player 2 metadata
        # Ensure consistent size regardless of active items
        player1_metadata = np.zeros(18, dtype=np.float32)
        player2_metadata = np.zeros(18, dtype=np.float32)

        # Populate player 1 metadata
        if self.player1_id is not None:
            p1_data = players_json[self.player1_id]
            if p1_data["body"]: # Check if body is not empty
                player1_metadata[0] = p1_data["body"][0]["row"] / map_height # Normalized head row
                player1_metadata[1] = p1_data["body"][0]["column"] / map_width # Normalized head column
            # Note: Your original code used move_names[players_json[0]["lastMoveDirection"]] = 1
            # This assumes a specific mapping and a fixed size array.
            # A more flexible approach might encode direction differently or include velocity.
            # For now, keeping a placeholder based on your structure:
            last_move = p1_data.get("lastMoveDirection")
            if last_move in move_names:
                 # Set the corresponding index to 1 (assuming one-hot encoding of direction)
                 player1_metadata[move_names[last_move] + 2] = 1 # +2 because 0 and 1 are head pos

            # Add active item information
            for item in p1_data.get("activeItems", []):
                 item_type = item.config["type"]
                 #item_type = item.get("type")
                 if item_type in items_names_to_players:
                     # Set the corresponding index to 1 for the item presence
                     player1_metadata[items_names_to_players[item_type]] = 1
                     # Set the next index for normalized duration
                     player1_metadata[items_names_to_players[item_type] + 1] = item.duration / 15.0 # Normalize duration

        # Populate player 2 metadata
        if self.player2_id is not None:
            p2_data = players_json[self.player2_id]
            if p2_data["body"]: # Check if body is not empty
                player2_metadata[0] = p2_data["body"][0]["row"] / map_height # Normalized head row
                player2_metadata[1] = p2_data["body"][0]["column"] / map_width # Normalized head column
            last_move = p2_data.get("lastMoveDirection")
            if last_move in move_names:
                 player2_metadata[move_names[last_move] + 2] = 1 # +2 because 0 and 1 are head pos

            # Add active item information
            for item in p2_data.get("activeItems", []):
                 item_type = item.config["type"]
                 #item_type = item.get("type")
                 if item_type in items_names_to_players:
                     player2_metadata[items_names_to_players[item_type]] = 1
                     player2_metadata[items_names_to_players[item_type] + 1] = item.duration / 15.0 # Normalize duration

        # Combine player metadata and add game-level metadata
        # Correctly concatenate numerical arrays
        move_count = game_info.get("moveCount", 0)
        game_metadata = np.array([move_count / 900.0], dtype=np.float32) # Normalize move count
        points_diff = np.array([game_info["players"][self.player1_id]["score"] - game_info["players"][self.player2_id]["score"]] , dtype=np.float32)/10000

        metadata_array = np.concatenate((player1_metadata, player2_metadata, game_metadata, points_diff))

        # Ensure map_array has channels first for PyTorch CNN (C, H, W)
        map_array = np.transpose(map_array, (2, 0, 1))

        return  map_array,metadata_array # Also return the raw data for reward calculation

    def remember(self, data, data_novi):
        map_state, metadata_state = self.get_state(data) 
        action  = self.last_action
        reward = self.give_reward(data = data_novi, data_prosli= data)
        reward = reward
        next_map_state, next_metadata_state = self.get_state(data_novi)
        done = 1 if data_novi.get("winner") is not None else 0
        """
        Stores an experience tuple in the agent's memory.

        Args:
            map_state (np.ndarray): The current map state.
            metadata_state (np.ndarray): The current metadata state.
            action (list): The action taken (one-hot encoded).
            reward (float): The reward received.
            next_map_state (np.ndarray): The next map state.
            next_metadata_state (np.ndarray): The next metadata state.
            done (bool): Whether the episode is finished.

        """
        # Store the experience as a tuple of numpy arrays and other data
        self.memory.append((map_state, metadata_state, action, reward, next_map_state, next_metadata_state, done))

    def train_long_term(self):
        """Trains the model using a batch of experiences from the memory."""
        if self.n_games > 500 and isinstance(self.model, ResnetSnakeNN):
            self.model.freeze_backbone(False)

        
        if not len(self.memory) <1000 :
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
            

            # Unpack the samples into separate lists
            map_states, metadata_states, actions, rewards, next_map_states, next_metadata_states, dones = zip(*mini_sample)

            # Convert lists of numpy arrays to tensors
            map_states = torch.tensor(np.array(map_states), dtype=torch.float)
            metadata_states = torch.tensor(np.array(metadata_states), dtype=torch.float)
            actions = torch.tensor(np.array(actions), dtype=torch.long) # Actions are typically long
            rewards = torch.tensor(np.array(rewards), dtype=torch.float)
            next_map_states = torch.tensor(np.array(next_map_states), dtype=torch.float)
            next_metadata_states = torch.tensor(np.array(next_metadata_states), dtype=torch.float)
            dones = torch.tensor(np.array(dones), dtype=torch.bool) # Done flags are boolean

            # --- Call the trainer with separate inputs ---
            # You need to modify your QTrainer.train_step to accept these
            loss =self.trainer.train_step(map_states, metadata_states, actions, rewards, next_map_states, next_metadata_states, dones)
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
            
            self.model.eval()
            with torch.no_grad(): # No gradient calculation during inference
                prediction = self.model(map_state_tensor, metadata_state_tensor).cpu()
            # Set model back to training mode after inference
            self.model.train()
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

