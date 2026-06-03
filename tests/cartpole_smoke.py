import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "level0" / "cartpole"))

import gymnasium as gym

from q_logic_cartpole import catrpoleAgent


def main():
    agent = catrpoleAgent(gamma=0.99, double_q=True, priority=False, polyak=True)
    avg = 0
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    nonzero_losses = 0
    started_at = time.time()

    for game_no in range(episodes):
        env = gym.make("CartPole-v1")
        state, _ = env.reset(seed=42 + game_no)
        done = False
        count = 0
        loss = 0

        while not done:
            count += 1
            action = agent.get_action((state, done))
            next_state, _, terminated, truncated, _ = env.step(action)
            next_done = terminated or truncated
            agent.remember((state, done), (next_state, next_done))
            loss = agent.train()
            if loss:
                nonzero_losses += 1
            state = next_state
            done = next_done

        env.close()
        avg = count if game_no == 0 else 0.9 * avg + 0.1 * count
        print(f"episode={game_no} steps={count} avg={avg:.2f} last_loss={loss}")

    print(
        f"done episodes={episodes} memory={len(agent.memory)} "
        f"total_actions={agent.action_counter} train_steps={nonzero_losses} "
        f"seconds={time.time() - started_at:.2f}"
    )


if __name__ == "__main__":
    main()
