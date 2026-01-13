from Diplomski.q_logic.q_logic_pretraining import Pretrainer
import pretrain_probelms as pp
from pretraining_models import MultiHeadModel
from q_logic_snake import snakeAgent
from model_snake import DQNnoisy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.nn.functional import mse_loss as mse, binary_cross_entropy_with_logits as bce


def problem10(x):
    return pp.gt_tail_future(x,3)

def problem11(x):
    return pp.gt_tail_future(x,5)

def problem12(x):
    return pp.gt_tail_future(x,10)





class snake_pretrainer(Pretrainer):
    def __init__(self):
        problem_functions= [
            pp.problem1,
            pp.problem2,
            pp.problem3,
            pp.problem4,
            pp.problem5,
            pp.problem6,
            pp.problem7,
            pp.problem9,
            problem10,
            problem11,
            problem12
        ]
        output_shapes = [(2),(1),(8),(4),(4),(2),(2),(1),(2),(2),(2)]

        loss_functions = [mse,mse,mse,bce,mse,mse,mse,mse,mse,mse,mse]
        
        
        model = MultiHeadModel(output_shapes)
        optimizer = optim.Adam(model.parameters(),lr=5e-4)


        explore_model = DQNnoisy(map_channels=3, is_training=False)
        data = torch.load(r"C:\Users\lovro\Desktop\snake\model_saves\snake__polyakTrue_gamma0.99_doubleqTrue_priorityTrue_noisynetTrue_no_survive_reward_2025-12-01_14-33-12.pt.pt", map_location="cpu")
        explore_model.load_state_dict(data["model_state_dict"])
        possible_actions = [0,1,2,3]
        batch_size = 64


        super().__init__(explore_model, model, optimizer, possible_actions, batch_size, problem_functions, output_shapes, loss_functions,save_dir = "Diplomski/level2/classicSnake/model_saves")


    def get_explore_state(self, data):
        data,snake_state, reward, jabuka, done = data
        return {"x": torch.tensor(data, dtype=torch.float32)}

    def get_state(self, data):
        data,snake_state, reward, jabuka, done = data
        return {"x": torch.tensor(data, dtype=torch.float32)}
    
    def memory_to_model(self, memory_state):
        return memory_state

    def get_memory_state(self, data):
        return self.get_state(data)
    


class snake_pretrainer_head_position(snake_pretrainer):
    def __init__(self):
        super().__init__()

    def get_state(self, data):
        data,snake_state, reward, jabuka, done = data
        tenzor = np.zeros((4,data.shape[1],data.shape[2]))
        tenzor[:3,:,:] = data
        x,y = snake_state[0]
        tenzor[3,x,y] = 1 # head
        return {"x": torch.tensor(tenzor, dtype=torch.float32)}
    
class snake_pretrainer_head_tail(snake_pretrainer):
    def __init__(self):
        super().__init__()

    def get_state(self, data):
        data,snake_state, reward, jabuka, done = data
        tenzor = np.zeros((5,data.shape[1],data.shape[2]))
        tenzor[:3,:,:] = data
        x,y = snake_state[0]
        tenzor[3,x,y] = 1 # head
        x,y = snake_state[-1]
        tenzor[4,x,y] = 1 # tail
        return {"x": torch.tensor(tenzor, dtype=torch.float32)}
    
