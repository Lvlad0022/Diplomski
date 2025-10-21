import numpy as np
from q_logic_univerzalno import Agent
import torch.optim as optim
from simple_snake_models import SimpleSnakeNN, ResnetSnakeNN_Small, AdvancedSimpleSnakeNN
from loss_functions import huberPriorityLoss, huberLoss

move_names = {"up": 0, # Changed to 0, 1, 2 to match typical 3-action output (straight, right, left)
              "right": 1,
              "down": 2, # Assuming 'down' is same action as 'up' in terms of turning (straight)
              "left": 3} # Assuming 'left' is a turn left






class SimpleSnakeAgent(Agent):
    def __init__(self, train = True,n_step_remember=1, gamma=0.93, end_priority = False):
        model = SimpleSnakeNN()
        optimizer = optim.Adam(model.parameters(),lr=5e-4) 
        super().__init__(model = model, optimizer = optimizer, criterion= huberLoss(), train = train, n_step_remember=n_step_remember, end_priority=end_priority)  # pozove konstruktor od Agent
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
            if winner is not None:
                self.n_games += 1
                return 1.0 if winner else -1.0

            # living bonus
            reward = 0.005
            return reward


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