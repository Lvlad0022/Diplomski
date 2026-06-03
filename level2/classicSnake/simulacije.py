import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from environment import SimpleSnakeEnv
import time
import random
from q_logic_snake import snakeAgent


def main(model_path, episodes=100, board_size=10, sleep=0.03, render=True, noisy_net=True, model_type="modular_dueling_noisy", backbone_type="classic"):
    env = SimpleSnakeEnv(size = board_size)
    agent1 = snakeAgent(train=False, noisy_net=noisy_net, model_type=model_type, backbone_type=backbone_type)

    agent1.load_agent_state(model_path, noisynet=noisy_net, training=False)
    agent1.epsilon = 0
    agent1.is_training = False
    
    
    sum_jabuke = 0
    for i in range(episodes):
        state, snake = env.reset()
        done = False
        count = 0
        jabuka = 0
        reward = 0
        jabuke = 0
        while not done:
            count +=1
            # Random action just to view the game
            if(count <0):
                action = random.randint(0,3)
            else:
                action = agent1.get_action((state,snake,reward,jabuka,done))

            if reward >= 0.5:
                sum_jabuke += 1
                jabuke += 1
            state, snake, reward,done, info = env.step(action)
            if render:
                env.render()
            if sleep:
                time.sleep(sleep)
        print(jabuke)
        if render:
            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--board-size", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-noisy-net", action="store_true")
    parser.add_argument("--backbone", choices=["classic", "resnext_snake"], default="classic")
    parser.add_argument(
        "--model-type",
        choices=[
            "modular_classic",
            "modular_noisy",
            "modular_dueling",
            "modular_dueling_noisy",
        ],
        default="modular_dueling_noisy",
    )
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        episodes=args.episodes,
        board_size=args.board_size,
        sleep=args.sleep,
        render=not args.no_render,
        noisy_net=not args.no_noisy_net,
        model_type=args.model_type,
        backbone_type=args.backbone,
    )
