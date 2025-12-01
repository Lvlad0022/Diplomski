import torch.optim as optim
import torch
import numpy as np
from q_logic.q_logic_popravljeno import Agent
from math import cos

from model_snake import DQN, DQNnoisy
from q_logic.loss_functions import huberLoss
from q_logic.q_logic_memory_classes import TDPriorityReplayBuffer, ReplayBuffer
from q_logic.q_logic_schedulers import WarmupPeakDecayScheduler



class snakeAgent(Agent):
    def __init__(self, train = True,n_step_remember=1,  gamma=0.93, priority = False, memory = 0, advanced_logging_path= False, time_logging_path = False, model = None, double_q = False, polyak = True, noisy_net = False):
        self.reward_policy = True
        model =DQNnoisy(is_training=True) if noisy_net else DQN()
         
        optimizer = optim.Adam(model.parameters(),lr=5e-4)

        memory = ReplayBuffer()
        if priority:
            memory = TDPriorityReplayBuffer()

        possible_actions = [0,1,2,3]
                
        super().__init__(model = model, polyak_update=polyak, gamma = gamma, optimizer = optimizer, advanced_logging_path= advanced_logging_path,possible_actions =possible_actions,
                         criterion= huberLoss(), train = train, n_step_remember=n_step_remember, memory=memory, batch_size=64, double_q = double_q, noisy_net=noisy_net)  # pozove konstruktor od Agent
        

    def give_reward(self, data_novi, data, akcija):
        data_novi, reward, jabuka,done = data_novi
        if self.reward_policy == True:
            if done:
                reward = -1
            elif jabuka == 50:
                done = True
                reward = 1
            elif reward < 1:
                reward = -0.05
            else:
                reward = 1
        return reward, done

    def get_state(self, data):
        data, reward, jabuka, done = data
        return {"x": torch.tensor(np.array(data))}
    
    def memory_to_model(self, memory_state):
        return memory_state

    def get_memory_state(self, data):
        return self.get_state(data)
        