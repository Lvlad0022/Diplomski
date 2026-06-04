import argparse
import time
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from environment import CirclingRareChannelEnv


def clockwise_heuristic_action(env):
    row, col = env.position
    center_row, center_col = env.center

    # Tangent direction for clockwise movement around the center in screen coords.
    dy = row - center_row
    dx = col - center_col
    tangent_row = dx
    tangent_col = -dy

    scored_actions = []
    for action, (dr, dc) in env.ACTIONS.items():
        candidate = (row + dr, col + dc)
        if env._is_blocked(candidate):
            continue
        score = dr * tangent_row + dc * tangent_col
        scored_actions.append((score, action))

    if not scored_actions:
        return 0

    scored_actions.sort(reverse=True)
    return scored_actions[0][1]


def main(steps=200, sleep=0.15, random_actions=False, seed=7, channel=None):
    env = CirclingRareChannelEnv(seed=seed)
    _, info = env.reset(channel=channel)

    print("Initial state")
    env.render()

    for step in range(steps):
        if random_actions:
            action = env.rng.choice(list(env.ACTIONS.keys()))
        else:
            action = clockwise_heuristic_action(env)

        _, reward, done, info = env.step(action)
        print("")
        print(f"step={step + 1} action={action} reward={reward:.3f}")
        env.render()

        if done:
            print("")
            print("Episode done. Resetting environment.")
            _, info = env.reset(channel=channel)
            env.render()

        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channel", type=int, default=None)
    args = parser.parse_args()

    main(
        steps=args.steps,
        sleep=args.sleep,
        random_actions=args.random_actions,
        seed=args.seed,
        channel=args.channel,
    )
