import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
SNAKE_DIR = ROOT / "level2" / "classicSnake"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SNAKE_DIR))

from model_snake import build_snake_model
from q_logic.loss_functions import WeightedMSELoss, huberLoss
from q_logic.q_logic import Agent
from q_logic.q_logic_memory_classes import ReplayBuffer


def build_loss(loss_type):
    if loss_type == "mse":
        return WeightedMSELoss()
    if loss_type == "huber":
        return huberLoss(delta=1.0)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


class CirclingAgent(Agent):
    def __init__(
        self,
        train=True,
        gamma=0.97,
        lr=1e-4,
        batch_size=64,
        memory_capacity=100_000,
        n_step_remember=1,
        loss_type="mse",
        backbone_type="classic",
        double_q=True,
        polyak=True,
        noisy_net=True,
        save_dir="model_saves",
    ):
        self.noisy_net_enabled = noisy_net
        model = build_snake_model(
            backbone_type=backbone_type,
            head_type="dueling_noisy" if noisy_net else "dueling",
            map_channels=2,
            num_actions=4,
        )
        optimizer = optim.Adam(model.parameters(), lr=lr)
        memory = ReplayBuffer(capacity=memory_capacity)

        super().__init__(
            model=model,
            optimizer=optimizer,
            possible_actions=[0, 1, 2, 3],
            batch_size=batch_size,
            criterion=build_loss(loss_type),
            scheduler=False,
            train=train,
            double_q=double_q,
            n_step_remember=n_step_remember,
            gamma=gamma,
            memory=memory,
            save_dir=save_dir,
            polyak_update=polyak,
            noisy_net=noisy_net,
        )

    def give_reward(self, data_novi, data, akcija):
        return data_novi[1], data_novi[2]

    def get_state(self, data):
        obs = data[0] if isinstance(data, tuple) else data
        return self._obs_to_state(obs)

    def get_memory_state(self, data):
        return self.get_state(data)

    def memory_to_model(self, memory_state):
        return memory_state

    def _obs_to_state(self, obs):
        if isinstance(obs, torch.Tensor):
            tensor = obs.detach().clone().float()
        else:
            tensor = torch.tensor(np.array(obs), dtype=torch.float32)
        return {"x": tensor}
