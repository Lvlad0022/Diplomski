import torch.optim as optim
import torch
import numpy as np
from q_logic.q_logic_parallel import Agent_inference, Agent_trainer
from math import cos

from model_snake import DQN, DQNnoisy, DQNnoisy_attention, load_backbone_only, costum_model_load
from q_logic.loss_functions import huberLoss
from q_logic.q_logic_memory_classes import TDPriorityReplayBuffer, ReplayBuffer
from q_logic.q_logic_schedulers import LinearDecayScheduler, CosineAnealSchedulerWarmReset



class snakeAgent_inference(Agent_inference):
    def __init__(self, train = True,n_step_remember=1,  gamma=0.93, model = None, noisy_net = True, attention = False, on_gpu = False
                  ):
        self.reward_policy = True
        if model is not None:
            model = model
        else:
            if attention:
                model = DQNnoisy_attention(is_training= True)
            else:
                model =DQNnoisy(is_training=True) if noisy_net else DQN()
         

        possible_actions = [0,1,2,3]
                
        super().__init__(model,  possible_actions , train = train, on_gpu = on_gpu,
                        n_step_remember=n_step_remember, gamma=gamma, noisy_net = noisy_net)  # pozove konstruktor od Agent
        

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


class snakeAgent_trainer(Agent_trainer):
    def __init__(self, train = True,  gamma=0.93, model = None, noisy_net = True, attention = False, on_gpu = True,
                 polyak = True, double_q = True, advanced_logging_path= False):
        self.reward_policy = True
        if model is not None:
            model = model
        else:
            model =DQNnoisy(is_training=True) if noisy_net else DQN()
        
                 
        optimizer = optim.Adam(model.parameters(),lr=1e-4)
        scheduler = CosineAnealSchedulerWarmReset(optimizer )


        possible_actions = [0,1,2,3]
                
        super().__init__(model = model, polyak_update=polyak, gamma = gamma, optimizer = optimizer, scheduler=scheduler, advanced_logging_path= advanced_logging_path,possible_actions =possible_actions,
                         criterion= huberLoss(), train = train, batch_size=32, double_q = double_q, noisy_net=noisy_net)  # pozove konstruktor od Agent
        

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