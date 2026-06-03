import torch.optim as optim
import torch
import numpy as np
from q_logic.q_logic import Agent

from model_snake import build_snake_model
from q_logic.loss_functions import WeightedMSELoss, huberLoss
from q_logic.q_logic_memory_classes import (
    ReplayBuffer,
    TDDecayPriorityReplayBuffer,
    TDMixPriorityReplayBuffer,
    TDPriorityReplayBuffer,
)
from q_logic.q_logic_schedulers import CosineAnealSchedulerWarmReset, CosineWarmupHoldDecayScheduler


MODEL_TYPE_TO_HEAD = {
    "modular_classic": ("classic", False),
    "modular_noisy": ("noisy", True),
    "modular_dueling": ("dueling", False),
    "modular_dueling_noisy": ("dueling_noisy", True),
}


def build_replay_buffer(replay_buffer_type, priority_decay=0.995, td_mix=0.5):
    if replay_buffer_type == "uniform":
        return ReplayBuffer()
    if replay_buffer_type == "td":
        return TDPriorityReplayBuffer()
    if replay_buffer_type == "td_decay":
        return TDDecayPriorityReplayBuffer(priority_decay=priority_decay)
    if replay_buffer_type == "td_mix":
        return TDMixPriorityReplayBuffer(td_mix=td_mix)
    raise ValueError(f"Unsupported replay_buffer_type: {replay_buffer_type}")


def build_loss(loss_type):
    if loss_type == "huber":
        return huberLoss(delta=1.0)
    if loss_type == "mse":
        return WeightedMSELoss()
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def build_scheduler(
    scheduler_type,
    optimizer,
    scheduler_warmup_steps=5_000,
    scheduler_hold_steps=5_000,
    scheduler_decay_steps=500_000,
    scheduler_initial_lr=1e-4,
    scheduler_max_lr=5e-4,
    scheduler_final_lr=1e-6,
):
    if scheduler_type == "warm_restart":
        return CosineAnealSchedulerWarmReset(optimizer)
    if scheduler_type == "cosine_warmup_hold":
        return CosineWarmupHoldDecayScheduler(
            optimizer,
            warmup_steps=scheduler_warmup_steps,
            hold_steps=scheduler_hold_steps,
            decay_steps=scheduler_decay_steps,
            initial_lr=scheduler_initial_lr,
            max_lr=scheduler_max_lr,
            final_lr=scheduler_final_lr,
        )
    raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")



class snakeAgent(Agent):
    def __init__(self, train = True,n_step_remember=1,  gamma=0.93, priority = True, memory = 0, advanced_logging_path= False,
                  time_logging_path = False, model = None, double_q = False, polyak = True, noisy_net = False,
                  save_dir = "model_saves", model_type = "modular_dueling_noisy", backbone_type = "classic",
                  replay_buffer_type = None, priority_decay=0.995, td_mix=0.5, loss_type="huber",
                  scheduler_type="warm_restart", scheduler_warmup_steps=5_000, scheduler_hold_steps=5_000,
                  scheduler_decay_steps=500_000, scheduler_initial_lr=1e-4, scheduler_max_lr=5e-4,
                  scheduler_final_lr=1e-6,
                  ):
        self.reward_policy = True
        if model is not None:
            model = model
        else:
            try:
                head_type, noisy_net = MODEL_TYPE_TO_HEAD[model_type]
            except KeyError as exc:
                raise ValueError(f"Unsupported model_type: {model_type}") from exc
            model = build_snake_model(backbone_type=backbone_type, head_type=head_type)
         
        optimizer = optim.Adam(model.parameters(), lr=scheduler_initial_lr)
        scheduler = build_scheduler(
            scheduler_type,
            optimizer,
            scheduler_warmup_steps=scheduler_warmup_steps,
            scheduler_hold_steps=scheduler_hold_steps,
            scheduler_decay_steps=scheduler_decay_steps,
            scheduler_initial_lr=scheduler_initial_lr,
            scheduler_max_lr=scheduler_max_lr,
            scheduler_final_lr=scheduler_final_lr,
        )

        if memory == 0:
            if replay_buffer_type is None:
                replay_buffer_type = "td" if priority else "uniform"
            memory = build_replay_buffer(
                replay_buffer_type,
                priority_decay=priority_decay,
                td_mix=td_mix,
            )

        possible_actions = [0,1,2,3]
        criterion = build_loss(loss_type)
                
        super().__init__(model = model, polyak_update=polyak, gamma = gamma, optimizer = optimizer, scheduler=scheduler, advanced_logging_path= advanced_logging_path,possible_actions =possible_actions,
                         criterion= criterion, train = train, n_step_remember=n_step_remember, memory=memory, batch_size=64, double_q = double_q, noisy_net=noisy_net, save_dir=save_dir)  # pozove konstruktor od Agent
        

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
    def __init__(self, train = True,n_step_remember=1,  gamma=0.93, priority = False, memory = 0, advanced_logging_path= False, time_logging_path = False, model = None, double_q = False, polyak = True, noisy_net = False, save_dir = "model_saves", model_type = "modular_dueling_noisy", backbone_type = "classic", replay_buffer_type = None, priority_decay=0.995, td_mix=0.5, loss_type="huber", scheduler_type="warm_restart", scheduler_warmup_steps=5_000, scheduler_hold_steps=5_000, scheduler_decay_steps=500_000, scheduler_initial_lr=1e-4, scheduler_max_lr=5e-4, scheduler_final_lr=1e-6):
        if model is None:
            try:
                head_type, noisy_net = MODEL_TYPE_TO_HEAD[model_type]
            except KeyError as exc:
                raise ValueError(f"Unsupported model_type: {model_type}") from exc
            model = build_snake_model(backbone_type=backbone_type, head_type=head_type, map_channels=4)
        super().__init__(train, n_step_remember, gamma, priority, memory, advanced_logging_path,time_logging_path, model, double_q,polyak,noisy_net, save_dir, model_type, backbone_type, replay_buffer_type, priority_decay, td_mix, loss_type, scheduler_type, scheduler_warmup_steps, scheduler_hold_steps, scheduler_decay_steps, scheduler_initial_lr, scheduler_max_lr, scheduler_final_lr)  # pozove konstruktor od Agent
        
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
