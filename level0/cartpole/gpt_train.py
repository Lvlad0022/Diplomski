import torch
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from gpt_dqn import DQN , ReplayBuffer
import time
from q_logic.q_logic_logging import CSVLogger, make_run_name


env = gym.make("CartPole-v1")
q = DQN().to("cuda")
target_q = DQN().to("cuda")
target_q.load_state_dict(q.state_dict())

optimizer = optim.Adam(q.parameters(), lr=1e-3)
buffer = ReplayBuffer()

gamma = 0.99
batch_size = 64
epsilon = 1.0
eps_decay = 0.995
eps_min = 0.05
target_update_freq = 1000

state, _ = env.reset()
global_step = 0

file_name = make_run_name(f"carptpole_gpt")
logger = CSVLogger(file_name, fieldnames=[
                    "game", "avg_count", "vrijeme"  ])

vrijeme_avg = 0
avg_count = 10
for episode in range(6000):
    done = False
    state, _ = env.reset()
    episode_reward = 0
    a = time.time()
    count = 0
    while not done:
        count += 1
        global_step += 1

        # --- Epsilon-greedy action ---
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).to("cuda")
                action = q(s).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        # --- Learning step ---
        if len(buffer) > batch_size:
            s, a, r, ns, d = buffer.sample(batch_size)

            s = torch.tensor(s, dtype=torch.float32).to("cuda")
            ns = torch.tensor(ns, dtype=torch.float32).to("cuda")
            a = torch.tensor(a, dtype=torch.long).to("cuda")
            r = torch.tensor(r, dtype=torch.float32).to("cuda")
            d = torch.tensor(d, dtype=torch.float32).to("cuda")

            # Q(s,a)
            q_values = q(s).gather(1, a.unsqueeze(1)).squeeze(1)

            # target: r + gamma * max_a' Q_target(ns)
            with torch.no_grad():
                next_q = target_q(ns).max(1)[0]
                target = r + gamma * next_q * (1 - d)

            loss = F.mse_loss(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Epsilon decay ---
        epsilon = max(eps_min, epsilon * eps_decay)

        # --- Target network sync ---
        if global_step % target_update_freq == 0:
            target_q.load_state_dict(q.state_dict())
    vrijeme = time.time()-a
    vrijeme_avg = vrijeme_avg*0.99 + vrijeme*0.01
    avg_count = avg_count*0.99 + count*0.01
    print(f"Episode {episode}, reward={episode_reward}, eps={epsilon:.3f}")

    logger.log({
                "game": episode,
                "avg_count": avg_count,
                "vrijeme": vrijeme_avg
            })