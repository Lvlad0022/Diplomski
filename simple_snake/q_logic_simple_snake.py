import torch.optim as optim
import numpy as np
from q_logic.q_logic_univerzalno import Agent

from simple_snake_models import SimpleSnakeNN, DuelingSimpleSnakeNN
from q_logic.loss_functions import huberLoss
from q_logic.q_logic_memory_classes import ReplayBuffer, RewardPriorityReplayBuffer, TDPriorityReplayBuffer, TDPriorityReplayBuffer_2
from q_logic.q_logic_schedulers import TDAdaptiveScheduler, LossAdaptiveLRScheduler, WarmupPeakDecayScheduler

move_names = {"up": 0, # Changed to 0, 1, 2 to match typical 3-action output (straight, right, left)
              "right": 1,
              "down": 2, # Assuming 'down' is same action as 'up' in terms of turning (straight)
              "left": 3} # Assuming 'left' is a turn left






class SimpleSnakeAgent(Agent):
    def __init__(self, train = True,n_step_remember=1, snake_i="nondueling", scheduler= 0  ,gamma=0.93, end_priority = 1, memory = 0, advanced_logging_path= False, time_logging_path = False, model = None):
        if model is None:
            model = SimpleSnakeNN() if snake_i == "nondueling" else DuelingSimpleSnakeNN()
        else:
            model = model
        optimizer = optim.Adam(model.parameters(),lr=5e-4)# optimizer se uvijek mora poslati scheduleru 


        if scheduler == 1:
            scheduler = TDAdaptiveScheduler(optimizer)
        elif scheduler == 2:
            scheduler = LossAdaptiveLRScheduler(optimizer)
        elif scheduler == 3:
            scheduler = WarmupPeakDecayScheduler(optimizer)
        elif scheduler == 4:
            scheduler = WarmupPeakDecayScheduler(optimizer, initial_lr=1e-5, max_lr=1e-4, final_lr=1e-10)
        



        
        if memory == 0:
             memory = ReplayBuffer(n_step_remember =n_step_remember)
        if memory == 1: 
             memory = TDPriorityReplayBuffer(n_step_remember =n_step_remember)
        if memory == 2: 
             memory = TDPriorityReplayBuffer(n_step_remember =n_step_remember, weights = False, predecesor=True)
        if memory == 3:
             memory = TDPriorityReplayBuffer(n_step_remember =n_step_remember, weights = False, predecesor=False)
        if memory == 4:
             memory = TDPriorityReplayBuffer(n_step_remember =n_step_remember, weights = False, predecesor=False)
        if memory == 5:
             memory = TDPriorityReplayBuffer(n_step_remember =n_step_remember, weights = True, predecesor=True,)
        if memory == 6: 
             memory = TDPriorityReplayBuffer_2(n_step_remember =n_step_remember)
        
        super().__init__(model = model, optimizer = optimizer, scheduler=scheduler, advanced_logging_path= advanced_logging_path, time_logging_path = time_logging_path,
                         criterion= huberLoss(), train = train, n_step_remember=n_step_remember, memory=memory)  # pozove konstruktor od Agent
        print("SimpleSnakeAgent initialized!")
  
    def divide_lr(self,mult=3):
        
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] /= mult
            print(f"Learning rate changed to: {param_group['lr']}")

    def change_lr(self,lr):
        
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] = lr
            print(f"Learning rate changed to: {param_group['lr']}")


    def give_reward(self, data_novi, data, akcija):
            winner = data_novi.get("winner")
            done = 0
            reward = 0.05
            if winner is not None:
                self.n_games += 1
                done = 1
                reward = 0.05 if winner else -1.0
            
            return reward,done


    def get_state(self, data):
            map_height = 25
            map_width = 60
            map_json = data["map"]
            players_json = data["players"]
            game_info = data 
            
            map_array = np.zeros((map_height, map_width, 2), dtype=np.float32) # Use float32 for torch


            # Populate the map array
            for i, row in enumerate(map_json):
                for j, item in enumerate(row):
                    if item is not None:
                        item_type = item.get("type") # Use .get() for safety
                        if item_type == "border":
                            map_array[i, j, 0] = 1  # Border channel
                        elif item_type in ["snake-body", "snake-head"]:
                            map_array[i, j, 1] = 1 if item_type == "snake-head" else 0.5 # Your snake channel

            # --- Metadata Processing ---
            # Create arrays for player 1 and player 2 metadata
            # Ensure consistent size regardless of active items
            player_metadata = np.zeros(6, dtype=np.float32)

            p1_data = players_json[0]
            if p1_data["body"]: # Check if body is not empty
                player_metadata[0] = p1_data["body"][0]["row"] / map_height # Normalized head row
                player_metadata[1] = p1_data["body"][0]["column"] / map_width # Normalized head column
            # Note: Your original code used move_names[players_json[0]["lastMoveDirection"]] = 1
            # This assumes a specific mapping and a fixed size array.
            # A more flexible approach might encode direction differently or include velocity.
            # For now, keeping a placeholder based on your structure:
            last_move = p1_data.get("lastMoveDirection")
            if last_move in move_names:
                    # Set the corresponding index to 1 (assuming one-hot encoding of direction)
                    player_metadata[move_names[last_move] + 2] = 1 # +2 because 0 and 1 are head pos

            # Combine player metadata and add game-level metadata
            # Correctly concatenate numerical arrays
            move_count = game_info.get("moveCount", 0)
            game_metadata = np.array([move_count / 900.0], dtype=np.float32) # Normalize move count

            metadata_array = np.concatenate((player_metadata, game_metadata))

            # Ensure map_array has channels first for PyTorch CNN (C, H, W)
            map_array = np.transpose(map_array, (2, 0, 1))

            return  map_array,metadata_array # Also return the raw data for reward calculation
    
    def episode_count(self, metadata_states):
        state_episode_count = np.zeros((len(metadata_states,)))
        for i,state in enumerate(metadata_states):
            state_episode_count[i] = state[-1] * 900

        return
    


class SimpleSnakeAgent2(SimpleSnakeAgent):
    def __init__(self, train = True,n_step_remember=1, snake_i="nondueling", scheduler= 0  ,gamma=0.93, end_priority = 1, memory = 0, advanced_logging_path= False, time_logging_path = False):
        super().__init__(train = train, n_step_remember=n_step_remember, snake_i=snake_i, scheduler=scheduler, gamma=gamma, end_priority=end_priority,memory=memory, advanced_logging_path=advanced_logging_path, time_logging_path=time_logging_path)

    def get_state(self, data):
        map_height = 25
        map_width = 60
        map_json = data["map"]
        players_json = data["players"]
        game_info = data 
        
        map_array = np.zeros((map_height, map_width, 2), dtype=np.float32) # Use float32 for torch


        # Populate the map array
        for i, row in enumerate(map_json):
            for j, item in enumerate(row):
                if item is not None:
                    item_type = item.get("type") # Use .get() for safety
                    if item_type == "border":
                        map_array[i, j, 0] = 1  # Border channel
                    elif item_type in ["snake-body", "snake-head"]:
                        map_array[i, j, 1] = -1 if item_type == "snake-head" else 1 # Your snake channel

        # --- Metadata Processing ---
        # Create arrays for player 1 and player 2 metadata
        # Ensure consistent size regardless of active items
        player_metadata = np.zeros(6, dtype=np.float32)

        p1_data = players_json[0]
        if p1_data["body"]: # Check if body is not empty
            player_metadata[0] = p1_data["body"][0]["row"] / map_height # Normalized head row
            player_metadata[1] = p1_data["body"][0]["column"] / map_width # Normalized head column
        # Note: Your original code used move_names[players_json[0]["lastMoveDirection"]] = 1
        # This assumes a specific mapping and a fixed size array.
        # A more flexible approach might encode direction differently or include velocity.
        # For now, keeping a placeholder based on your structure:
        last_move = p1_data.get("lastMoveDirection")
        if last_move in move_names:
                # Set the corresponding index to 1 (assuming one-hot encoding of direction)
                player_metadata[move_names[last_move] + 2] = 1 # +2 because 0 and 1 are head pos

        # Combine player metadata and add game-level metadata
        # Correctly concatenate numerical arrays
        move_count = game_info.get("moveCount", 0)
        game_metadata = np.array([move_count / 900.0], dtype=np.float32) # Normalize move count

        metadata_array = np.concatenate((player_metadata, game_metadata))

        # Ensure map_array has channels first for PyTorch CNN (C, H, W)
        map_array = np.transpose(map_array, (2, 0, 1))

        return  map_array,metadata_array
    
class SimpleSnakeAgent3(SimpleSnakeAgent):
    def __init__(self, train = True,n_step_remember=1, snake_i="nondueling", scheduler= 0  ,gamma=0.93, end_priority = 1, memory = 0, advanced_logging_path= False, time_logging_path = False):
        model = SimpleSnakeNN(map_channels=3) if snake_i == "nondueling" else DuelingSimpleSnakeNN(map_channels=3)
        super().__init__(model= model, train = train, n_step_remember=n_step_remember, snake_i=snake_i, scheduler=scheduler, gamma=gamma, end_priority=end_priority,memory=memory, advanced_logging_path=advanced_logging_path, time_logging_path=time_logging_path)

    def get_state(self, data):
        map_height = 25
        map_width = 60
        map_json = data["map"]
        players_json = data["players"]
        game_info = data 
        
        map_array = np.zeros((map_height, map_width, 3), dtype=np.float32) # Use float32 for torch


        # Populate the map array
        for i, row in enumerate(map_json):
            for j, item in enumerate(row):
                if item is not None:
                    item_type = item.get("type") # Use .get() for safety
                    if item_type == "border":
                        map_array[i, j, 0] = 1  # Border channel
                    elif item_type in [ "snake-head"]:
                        map_array[i, j, 1] = 1
                    elif item_type in ["snake-body"]:
                        map_array[i, j, 2] = 1

        # --- Metadata Processing ---
        # Create arrays for player 1 and player 2 metadata
        # Ensure consistent size regardless of active items
        player_metadata = np.zeros(6, dtype=np.float32)

        p1_data = players_json[0]
        if p1_data["body"]: # Check if body is not empty
            player_metadata[0] = p1_data["body"][0]["row"] / map_height # Normalized head row
            player_metadata[1] = p1_data["body"][0]["column"] / map_width # Normalized head column
        # Note: Your original code used move_names[players_json[0]["lastMoveDirection"]] = 1
        # This assumes a specific mapping and a fixed size array.
        # A more flexible approach might encode direction differently or include velocity.
        # For now, keeping a placeholder based on your structure:
        last_move = p1_data.get("lastMoveDirection")
        if last_move in move_names:
                # Set the corresponding index to 1 (assuming one-hot encoding of direction)
                player_metadata[move_names[last_move] + 2] = 1 # +2 because 0 and 1 are head pos

        # Combine player metadata and add game-level metadata
        # Correctly concatenate numerical arrays
        move_count = game_info.get("moveCount", 0)
        game_metadata = np.array([move_count / 900.0], dtype=np.float32) # Normalize move count

        metadata_array = np.concatenate((player_metadata, game_metadata))

        # Ensure map_array has channels first for PyTorch CNN (C, H, W)
        map_array = np.transpose(map_array, (2, 0, 1))

        return  map_array,metadata_array