import torch.optim as optim
import torch
import numpy as np
from q_logic.q_logic import Agent
from math import cos

from model_snake import DQN, DQNnoisy, DQNnoisy_residual_backbone, load_backbone_only, costum_model_load
from q_logic.loss_functions import huberLoss
from q_logic.q_logic_memory_classes import TDPriorityReplayBuffer, ReplayBuffer
from q_logic.q_logic_schedulers import LinearDecayScheduler, CosineAnealSchedulerWarmReset



class snakeAgent(Agent):
    def __init__(self, train = True,n_step_remember=1,  gamma=0.93, priority = False, memory = 0, advanced_logging_path= False,
                  time_logging_path = False, model = None, double_q = False, polyak = True, noisy_net = False, residual = False
                  ):
        self.reward_policy = True
        if model is not None:
            model = model
        else:
            if residual:
                model = DQNnoisy_residual_backbone(is_training= True)
            else:
                model =DQNnoisy(is_training=True) if noisy_net else DQN()
         
        optimizer = optim.Adam(model.parameters(),lr=1e-6)
        #scheduler = CosineAnealSchedulerWarmReset(optimizer)
        scheduler = None

        memory = ReplayBuffer()
        if priority:
            memory = TDPriorityReplayBuffer()

        possible_actions = [0,1,2,3]
                
        super().__init__(model = model, polyak_update=polyak, gamma = gamma, optimizer = optimizer, scheduler=scheduler, advanced_logging_path= advanced_logging_path,possible_actions =possible_actions,
                         criterion= huberLoss(), train = train, n_step_remember=n_step_remember, memory=memory, batch_size=64, double_q = double_q, noisy_net=noisy_net)  # pozove konstruktor od Agent
        

    def give_reward(self, data_novi, data, akcija):
        data_novi, snake_state,reward, jabuka,done = data_novi
        if self.reward_policy:
            if done:
                reward = -1
            elif jabuka == 50:
                done = True
                reward = 1
            elif reward < 1:
                reward = 0
            else:
                reward = 1
        return reward, done

    def get_state(self, data):
        data, snake_state, reward, jabuka, done = data
        return {"x": torch.tensor(np.array(data))}
    
    def memory_to_model(self, memory_state):
        return memory_state

    def get_memory_state(self, data):
        return self.get_state(data)





class snakeAgent_head(snakeAgent):
    def __init__(self, train = True,n_step_remember=1,  gamma=0.93, priority = False, memory = 0, advanced_logging_path= False, time_logging_path = False, model = None, double_q = False, polyak = True, noisy_net = False):
        model = model =DQNnoisy(is_training=True, map_channels=4) if noisy_net else DQN(map_channels=4)
        super().__init__(train, n_step_remember, gamma, priority, memory, advanced_logging_path,time_logging_path, model, double_q,polyak,noisy_net)  # pozove konstruktor od Agent
        
    def get_state(self, data):
        data, snake_state, reward, jabuka, done = data
        a = np.array(data)
        out = np.zeros((4,a.shape[1],a.shape[2]))
        x,y = snake_state[0]
        out[:3,:,:] = a
        out[3,x,y] = 1 # head
        return {"x": torch.tensor(np.array(out, dtype= np.float32))}
    

    

"""
self.backbone_params = []
        self.head_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                self.backbone_params.append(param)
            else:
                self.head_params.append(param)
        optimizer = optim.Adam([
            {'params': self.backbone_params, 'lr': 5e-5},  
            {'params': self.head_params, 'lr': 5e-4}      
        ])
"""