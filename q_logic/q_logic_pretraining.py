
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

class MemoryBuffer:
    def __init__(self, num_problems, capacity = 10_000):
        self.capacity = capacity
        self.memory_solution =  [[] for i in range(num_problems)]
        self.memory_state = []
        self.counter = 0
    
    def push(self, state,solution):
        if self.counter < self.capacity:
            for d,list in zip(solution,self.memory_solution):
                list.append(d)
            self.memory_state.append(state)
            self.counter += 1
        else:
            for d,list in zip(solution,self.memory_solution):
                list[self.counter % self.capacity] = d
            self.memory_state[self.counter % self.capacity] = state
            self.counter += 1
                
    def sample(self, batch_size):
        sample_index = np.random.choice(min(self.counter, self.capacity), size=batch_size, replace=False)
        sample_solution = [[m[i] for i in sample_index] for m in self.memory_solution]
        sample_state = [self.memory_state[i] for i in sample_index]

        return sample_state, sample_solution
    
    def __len__(self):
        return len(self.memory_state)




class Pretrainer:
    def __init__(self, explore_model, model, optimizer, possible_actions ,batch_size,problem_functions, problem_solution_shapes, loss_functions,
                capacity= 25_000,  scheduler = False , save_dir = "model_saves"):
        
        
        self.save_dir = save_dir
        self.possible_actions = possible_actions
        self.num_actions = len(possible_actions)
        
        self.optimizer = optimizer
        self.scheduler = scheduler


        self.n_games = 0
        self.action_counter= 0
        self.batch_size = batch_size

        self.num_problems = len(problem_functions)
        self.problem_solution_shapes = problem_solution_shapes
        self.problem_functions = problem_functions
        self.memory = MemoryBuffer(self.num_problems, capacity=capacity)
        self.loss_functions = loss_functions

        #modeli
        self.Explore_model = explore_model
        self.model = model.to(DEVICE)

         


    def save_agent_state(self, file_name='agent_state.pth', training = False):
        os.makedirs(self.save_dir, exist_ok=True)
        file_path = f"{self.save_dir}/{file_name}.pt" 

        
        
        checkpoint  ={
            "model_state_dict": self.model.backbone.state_dict(),
        }

        torch.save(checkpoint, file_path)
        print("agent state saved")
        


    def return_counter(self):
        return self.counter

    def remember(self, data):
        state = self.get_state(data)
        solutions = [function(**state) for function in self.problem_functions]
        memory_state = self.get_memory_state(data)
        self.memory.push(memory_state,solutions)


    def train(self):

        if not len(self.memory) <1000 :
            sample_states, sample_solutions = self.memory.sample(self.batch_size)
            game_states = self.to_device(self.stack_state_batch(self.memory_to_model(sample_states))) #prvo prebacuje iz oblika za memmory u oblik za treniranje, pa uzima listu torch tenzora i prabacuje u jedan tako da spaja po axis 0 zatim prebacuje sve na cudu
            targets = [torch.tensor(np.stack(list, axis=0),dtype=torch.float32, device=DEVICE) for list in sample_solutions]



            self.model.train()

            outputs = self.model(**game_states) 
            losses = []
            losses_cpu = np.zeros((self.num_problems))
            for i, (pred, target, loss_f) in enumerate(zip(outputs, targets, self.loss_functions)):
                loss = loss_f(pred, target)
                losses_cpu[i] = loss.item()     
                losses.append(loss)    
            loss_total = sum(losses)            

            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            
            return np.sum(losses_cpu), losses_cpu
        return 0, 0 

    def episode_count(self, metadata_states):
        return np.zeros((len(metadata_states),))
    
    def get_action(self, data):
        self.action_counter += 1
        game_state = self.get_explore_state(data)
        
        final_move = np.zeros((self.num_actions,)) 

        self.Explore_model.eval()
        with torch.no_grad():
            prediction = self.Explore_model(**game_state).cpu()
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        move_direction = self.possible_actions[move]
            
            

        self.last_action = final_move
        return move_direction
    
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
