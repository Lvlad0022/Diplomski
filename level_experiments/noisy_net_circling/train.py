import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CURRENT_DIR))

from environment import CirclingRareChannelEnv
from q_logic.q_logic import set_seed
from q_logic.q_logic_logging import CSVLogger, make_run_name
from q_logic_circling import CirclingAgent


def make_agent_action(agent, obs):
    data = obs if isinstance(obs, tuple) else (obs, 0.0, False, {})
    if agent.noisy_net:
        action, _ = agent.get_action(data)
        return action
    return agent.get_action(data)


def main(
    num_steps=200_000,
    log_every=1_000,
    save_every=50_000,
    seed=42,
    loss_type="mse",
    backbone_type="classic",
    lr=1e-4,
    gamma=0.97,
    batch_size=64,
    memory_capacity=100_000,
    n_step_remember=3,
    slip_prob=0.0,
    clockwise_shaping_reward=0,
    save_model=True,
):
    set_seed(seed)

    run_name = make_run_name(
        f"circling_{backbone_type}_dueling_noisy_replayuniform_loss{loss_type}_nstep{n_step_remember}_lr{lr}_seed{seed}"
    )
    log_dir = CURRENT_DIR / "log"
    save_dir = CURRENT_DIR / "model_saves"

    logger = CSVLogger(
        str(log_dir / f"{run_name}.csv"),
        fieldnames=[
            "step",
            "avg_reward",
            "avg_loss",
            "avg_sectors",
            "lap_count",
            "lap_rate",
            "channel0_steps",
            "channel1_steps",
            "channel0_laps",
            "channel1_laps",
            "replay_size",
            "time",
        ],
    )

    env = CirclingRareChannelEnv(
        num_channels=2,
        common_channel_prob=0.95,
        max_steps=None,
        slip_prob=slip_prob,
        clockwise_shaping_reward=clockwise_shaping_reward,
        seed=seed,
    )
    agent = CirclingAgent(
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        memory_capacity=memory_capacity,
        n_step_remember=n_step_remember,
        loss_type=loss_type,
        backbone_type=backbone_type,
        double_q=True,
        polyak=True,
        noisy_net=True,
        save_dir=str(save_dir),
    )

    obs, info = env.reset()
    data = (obs, 0.0, False, info)
    channel_steps = np.zeros(2, dtype=np.int64)
    channel_laps = np.zeros(2, dtype=np.int64)
    reward_window = []
    loss_window = []
    sectors_window = []
    lap_count = 0
    start_time = time.time()

    print(
        f"Starting circling training: steps={num_steps}, backbone={backbone_type}, "
        f"loss={loss_type}, n_step={n_step_remember}, lr={lr}, gamma={gamma}, batch_size={batch_size}, "
        f"channel_probs={env.channel_probs.tolist()}, "
        f"clockwise_shaping_reward={clockwise_shaping_reward}, slip_prob={slip_prob}",
        flush=True,
    )

    for step in range(1, num_steps + 1):
        current_channel = info["channel"]
        action = make_agent_action(agent, data)
        next_obs, reward, done, next_info = env.step(action)
        next_data = (next_obs, reward, done, next_info)

        agent.remember(data, next_data)
        loss = agent.train()

        channel_steps[current_channel] += 1
        if next_info.get("lap_completed", False):
            lap_count += 1
            channel_laps[current_channel] += 1

        reward_window.append(float(reward))
        if loss:
            loss_window.append(float(loss))
        sectors_window.append(float(next_info["sectors_collected"]))

        data = next_data
        obs = next_obs
        info = next_info

        if step % log_every == 0:
            elapsed = time.time() - start_time
            avg_reward = float(np.mean(reward_window)) if reward_window else 0.0
            avg_loss = float(np.mean(loss_window)) if loss_window else 0.0
            avg_sectors = float(np.mean(sectors_window)) if sectors_window else 0.0
            lap_rate = lap_count / step

            row = {
                "step": step,
                "avg_reward": avg_reward,
                "avg_loss": avg_loss,
                "avg_sectors": avg_sectors,
                "lap_count": lap_count,
                "lap_rate": lap_rate,
                "channel0_steps": int(channel_steps[0]),
                "channel1_steps": int(channel_steps[1]),
                "channel0_laps": int(channel_laps[0]),
                "channel1_laps": int(channel_laps[1]),
                "replay_size": len(agent.memory),
                "time": elapsed,
            }
            logger.log(row)
            print(
                f"step={step} avg_reward={avg_reward:.4f} avg_loss={avg_loss:.5f} "
                f"avg_sectors={avg_sectors:.3f} laps={lap_count} "
                f"ch_steps={channel_steps.tolist()} ch_laps={channel_laps.tolist()} "
                f"replay={len(agent.memory)}",
                flush=True,
            )

            reward_window.clear()
            loss_window.clear()
            sectors_window.clear()

        if save_model and save_every and step % save_every == 0:
            agent.save_agent_state(f"{run_name}_step_{step}")
            print(f"saved checkpoint: {run_name}_step_{step}.pt", flush=True)

    if save_model:
        agent.save_agent_state(f"{run_name}_final")
        print(f"saved final model: {run_name}_final.pt", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=200_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--backbone", choices=["classic", "resnext_snake"], default="classic")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-capacity", type=int, default=100_000)
    parser.add_argument("--n-step-remember", type=int, default=3)
    parser.add_argument("--slip-prob", type=float, default=0.0)
    parser.add_argument("--clockwise-shaping-reward", type=float, default=0.02)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    main(
        num_steps=args.num_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        loss_type=args.loss,
        backbone_type=args.backbone,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        memory_capacity=args.memory_capacity,
        n_step_remember=args.n_step_remember,
        slip_prob=args.slip_prob,
        clockwise_shaping_reward=args.clockwise_shaping_reward,
        save_model=not args.no_save,
    )
