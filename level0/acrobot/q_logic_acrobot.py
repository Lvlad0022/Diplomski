import torch.optim as optim
import torch
import numpy as np
from q_logic.q_logic_popravljeno import Agent
from math import cos

from model_acrobot import DQN
from q_logic.loss_functions import huberLoss
from q_logic.q_logic_memory_classes import TDPriorityReplayBuffer, ReplayBuffer
from q_logic.q_logic_schedulers import WarmupPeakDecayScheduler



class acrobot_agent(Agent):
    def __init__(self, train = True,n_step_remember=1,  gamma=0.93, priority = False, memory = 0, advanced_logging_path= False, time_logging_path = False, model = None, double_q = False, polyak = True):
        model = DQN() 
        optimizer = optim.Adam(model.parameters(),lr=1e-3)

        memory = ReplayBuffer()
        if priority:
            memory = TDPriorityReplayBuffer()

        possible_actions = [0,1,2]
                
        super().__init__(model = model, polyak_update=polyak, gamma = gamma, optimizer = optimizer, advanced_logging_path= advanced_logging_path,possible_actions =possible_actions,
                         criterion= huberLoss(), train = train, n_step_remember=n_step_remember, memory=memory, batch_size=64, double_q = double_q)  # pozove konstruktor od Agent
        

    def give_reward(self, data_novi, data, akcija):
        data_novi, done = data_novi
        return -1 + 0.1*(-cos(data_novi[4]) - cos(data_novi[4] + data_novi[5])), done

    def get_state(self, data):
        data, done = data
        return {"x": torch.tensor(np.array(data))}
    
    def memory_to_model(self, memory_state):
        return memory_state

    def get_memory_state(self, data):
        return self.get_state(data)
        