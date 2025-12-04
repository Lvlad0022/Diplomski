import torch.optim as optim
import numpy as np
from Diplomski.q_logic.q_logic import Agent

from snake_models import SnakeNN, DuelingSnakeNN
from q_logic.loss_functions import huberLoss
from q_logic.q_logic_memory_classes import TDPriorityReplayBuffer, ReplayBuffer
from q_logic.q_logic_schedulers import WarmupPeakDecayScheduler

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
    "katana":4,
    "armour":6,
    "tron":8,
    "freeze":10,
    "loop":12, # Note: 'loop' was in your code, but 'tron' and 'freeze' were also in lasting_items and items_names_to_players. Double check these mappings.
    "golden-apple":14
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


class SimpleSnakeAgent(Agent):
    def __init__(self,player1_name, train = True,n_step_remember=1, snake_i="nondueling", scheduler= 0  ,gamma=0.93, end_priority = 1, memory = 0, advanced_logging_path= False, time_logging_path = False, model = None):
        self.map_channels = 15
        model = SnakeNN(map_channels=self.map_channels, metadata_dim=34) if snake_i == "nondueling" else DuelingSnakeNN(map_channels=self.map_channels, metadata_dim=34)
        optimizer = optim.Adam(model.parameters(),lr=5e-4)# optimizer se uvijek mora poslati scheduleru 
        
        if scheduler == 0:
            scheduler = WarmupPeakDecayScheduler(optimizer, max_lr=3e-4,decay_steps=25_000, final_lr=1e-5, warmup_steps=500, peak_steps=1000)
        if scheduler == 1:
            scheduler = WarmupPeakDecayScheduler(optimizer, max_lr=1e-4, final_lr=1e-6)
        
        if memory == 0:
             memory = ReplayBuffer(n_step_remember =n_step_remember)
        if memory == 1: 
             memory = TDPriorityReplayBuffer(n_step_remember =n_step_remember, segment=False)
        
        super().__init__(model = model, optimizer = optimizer, scheduler=scheduler, advanced_logging_path= advanced_logging_path, time_logging_path = time_logging_path,
                         criterion= huberLoss(), train = train, n_step_remember=n_step_remember, memory=memory, batch_size=64)  # pozove konstruktor od Agent
        print("SimpleSnakeAgent initialized!")
        self.player1_name = player1_name
        self.player2_name = None
        self.player1_id = None # Store player IDs
        self.player2_id = None
        self.enemy_cons = 1 # Placeholder, adjust based on game state if available

        

    def give_reward(self, data_novi, data, akcija):
            winner = data_novi.get("winner")
            done = 0
            if winner is not None:
                done = 1
                return 1.0 if data_novi.get('winner') == self.player1_name else -1.0, done

            # living bonus
            reward = 0.001

            # immediate scoring (difference)
            p1_now  = data_novi["players"][self.player1_id]["score"]
            p1_prev = data["players"][self.player1_id]["score"]
            p2_now  = data_novi["players"][self.player2_id]["score"]
            p2_prev = data["players"][self.player2_id]["score"]

            score_gain = (p1_now - p1_prev) /10
            reward += 0.0005 * score_gain   # <- easy to feel; tune 0.005–0.02
            if data["players"][self.player1_id].get("lastMoveDirection") == data_novi["players"][self.player1_id].get("lastMoveDirection"):
                reward -= 1

            # (optional) late-game pressure if behind
            move_count = data_novi.get("moveCount", 0)
            if move_count > 700:
                lead = (p1_now - p2_now)
                reward += 0.005 * np.tanh(lead / 10.0)

            return reward, done

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
            
            map_array = np.zeros((map_height, map_width, self.map_channels), dtype=np.float32) # Use float32 for torch


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
                                if  item_type == "snake-head":
                                    map_array[i, j, 1] = 1 
                                elif item_type == "snake-body":
                                    map_array[i, j, 2] = 1 
                            elif player_name == self.player2_name:
                                if  item_type == "snake-head":
                                    map_array[i, j, 3] = 1 
                                elif item_type == "snake-body":
                                    map_array[i, j, 4] = 1 
                        elif item_type in items_names:
                            map_array[i, j, 4+ items_names[item_type]] = 1

            # --- Metadata Processing ---
            # Create arrays for player 1 and player 2 metadata
            # Ensure consistent size regardless of active items
            player1_metadata = np.zeros(16, dtype=np.float32)
            player2_metadata = np.zeros(16, dtype=np.float32)

            # Populate player 1 metadata
            if self.player1_id is not None:
                p1_data = players_json[self.player1_id]
                last_move = p1_data.get("lastMoveDirection")
                if last_move in move_names:
                    # Set the corresponding index to 1 (assuming one-hot encoding of direction)
                    player1_metadata[move_names[last_move]] = 1

                for item in p1_data.get("activeItems", []):
                    item_type = item.config["type"]
                    if item_type in items_names_to_players:
                        player1_metadata[items_names_to_players[item_type]] = 1
                        # Set the next index for normalized duration
                        player1_metadata[items_names_to_players[item_type] + 1] = item.duration / 15.0 # Normalize duration

            if self.player2_id is not None:
                p2_data = players_json[self.player2_id]
                last_move = p2_data.get("lastMoveDirection")
                if last_move in move_names:
                    player2_metadata[move_names[last_move] ] = 1 
                for item in p2_data.get("activeItems", []):
                    item_type = item.config["type"]
                    if item_type in items_names_to_players:
                        player2_metadata[items_names_to_players[item_type]] = 1
                        player2_metadata[items_names_to_players[item_type] + 1] = item.duration / 15.0 


            move_count = game_info.get("moveCount", 0)
            game_metadata = np.array([move_count / 900.0], dtype=np.float32) # Normalize move count
            points_diff = np.array([game_info["players"][self.player1_id]["score"] - game_info["players"][self.player2_id]["score"]] , dtype=np.float32)/10000

            metadata_array = np.concatenate((player1_metadata, player2_metadata, points_diff,game_metadata))

            map_array = np.transpose(map_array, (2, 0, 1))

            return  map_array,metadata_array # Also return the raw data for reward calculation