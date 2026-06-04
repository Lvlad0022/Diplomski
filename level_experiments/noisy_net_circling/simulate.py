import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CURRENT_DIR))

from environment import CirclingRareChannelEnv
from q_logic_circling import CirclingAgent


def latest_model_path():
    model_dir = CURRENT_DIR / "model_saves"
    candidates = sorted(model_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .pt models found in {model_dir}")
    return candidates[0]


def select_action(agent, obs, deterministic=True):
    data = (obs, 0.0, False, {})

    if deterministic:
        was_training = agent.model.training
        agent.model.eval()
        agent.model.is_training = False
        with torch.no_grad():
            state = agent.to_device(agent.get_state(data))
            q_values = agent.model(**state).cpu()
        agent.model.train(was_training)
        agent.model.is_training = was_training
        return int(torch.argmax(q_values).item()), q_values.squeeze(0).numpy()

    action, _ = agent.get_action(data)
    return int(action), None


def main(
    model_path=None,
    steps=2000,
    sleep=0.15,
    channel=None,
    seed=42,
    loss_type="mse",
    backbone_type="classic",
    deterministic=True,
    slip_prob=0.0,
):
    if model_path is None:
        model_path = latest_model_path()
    else:
        model_path = Path(model_path)

    env = CirclingRareChannelEnv(
        num_channels=2,
        common_channel_prob=0.95,
        max_steps=None,
        slip_prob=slip_prob,
        seed=seed,
    )
    agent = CirclingAgent(
        train=False,
        loss_type=loss_type,
        backbone_type=backbone_type,
        noisy_net=True,
        save_dir=str(CURRENT_DIR / "model_saves"),
    )
    agent.load_agent_state(str(model_path), training=False, noisynet=True)
    agent.is_training = False

    obs, info = env.reset(channel=channel)
    print(f"Loaded model: {model_path}")
    print("Initial state")
    env.render()

    total_reward = 0.0
    laps = 0
    for step in range(1, steps + 1):
        action, q_values = select_action(agent, obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if info.get("lap_completed", False):
            laps += 1

        print("")
        q_text = "" if q_values is None else f" q={q_values.round(3).tolist()}"
        print(
            f"step={step} action={action} reward={reward:.3f} total_reward={total_reward:.3f} "
            f"laps={laps}{q_text}"
        )
        env.render()

        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--backbone", choices=["classic", "resnext_snake"], default="classic")
    parser.add_argument("--slip-prob", type=float, default=0.0)
    parser.add_argument("--stochastic-noisy", action="store_true")
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        steps=args.steps,
        sleep=args.sleep,
        channel=args.channel,
        seed=args.seed,
        loss_type=args.loss,
        backbone_type=args.backbone,
        deterministic=not args.stochastic_noisy,
        slip_prob=args.slip_prob,
    )
