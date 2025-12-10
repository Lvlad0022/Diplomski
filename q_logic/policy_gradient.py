import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # Shared body
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Policy head
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: (batch, state_dim)
        returns:
           logits: (batch, action_dim)
           values: (batch, 1)
        """
        features = self.shared(x)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return logits, values

    def act(self, state):
        """
        state: np.array or 1D tensor, shape (state_dim,)
        returns: action (int), log_prob (tensor), value (tensor)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0)  # (1, state_dim)

        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.squeeze(0), value.squeeze(0)
    



class A2CAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=3e-4,
        value_coef=0.5,
        entropy_coef=0.01,
        n_steps=5,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps

        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    @torch.no_grad()
    def select_action(self, state):
        action, log_prob, value = self.model.act(state)
        return action, log_prob, value

    def rollout(self, env, state):
        """
        Collects n_steps of experience (or until done).
        Returns everything needed to train.
        """
        states = []
        actions = []
        rewards = []
        dones = []
        values = []

        for step in range(self.n_steps):
            action, log_prob, value = self.select_action(state)

            next_state, reward, done, info = env.step(action)
            # If using gymnasium: obs, reward, terminated, truncated, info
            # terminated, truncated = done, info.get("truncated", False)
            # done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value.cpu().numpy())

            state = next_state

            if done:
                state = env.reset()
                break

        # Bootstrap value
        with torch.no_grad():
            if dones[-1]:
                next_value = 0.0
            else:
                if not isinstance(state, torch.Tensor):
                    s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    s_t = state.to(self.device).unsqueeze(0)
                _, v = self.model(s_t)
                next_value = v.squeeze(0).cpu().item()

        rollout_data = {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.bool_),
            "values": np.array(values, dtype=np.float32).squeeze(-1),
            "next_value": next_value,
            "last_state": state,   # for continuing training loop
            "last_done": dones[-1]
        }
        return rollout_data

    def compute_returns_and_advantages(self, rewards, dones, values, next_value):
        """
        rewards, dones, values: 1D numpy arrays of length T
        next_value: scalar for bootstrap
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        R = next_value

        for t in reversed(range(T)):
            if dones[t]:
                R = 0.0
            R = rewards[t] + self.gamma * R
            returns[t] = R

        advantages = returns - values
        return returns, advantages

    def update(self, rollout_data):
        states = torch.tensor(rollout_data["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(rollout_data["actions"], dtype=torch.int64, device=self.device)
        rewards = rollout_data["rewards"]
        dones = rollout_data["dones"]
        values = rollout_data["values"]
        next_value = rollout_data["next_value"]

        # Compute returns and advantages (numpy)
        returns, advantages = self.compute_returns_and_advantages(
            rewards, dones, values, next_value
        )

        # Convert to tensors
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # Forward pass again for gradient (we don't use stored values here)
        logits, value_preds = self.model(states)
        value_preds = value_preds.squeeze(-1)  # (T,)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Losses
        actor_loss = -(log_probs * advantages_t.detach()).mean()
        critic_loss = (returns_t - value_preds).pow(2).mean()
        entropy_loss = -entropy

        loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        info = {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
        }
        return info
