import argparse
import sys
from pathlib import Path

import gymnasium as gym
import time
import random

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from q_logic_cartpole import catrpoleAgent


def main(model_path, episodes=20, render=True, sleep=0.01):
    render_mode = "human" if render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)
    agent1 = catrpoleAgent(train= False)

    agent1.load_agent_state(model_path)
    agent1.epsilon = 0
    agent1.is_training = False

    for i in range(episodes):
        state, info = env.reset()
        done = False
        count = 0
        while not done:
            count +=1
            # Random action just to view the game
            if(count < -1):
                action = random.randint(0,0)
            else:
                action = agent1.get_action((state,done))

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if sleep:
                time.sleep(sleep)
        print(count)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.01)
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        episodes=args.episodes,
        render=not args.no_render,
        sleep=args.sleep,
    )
