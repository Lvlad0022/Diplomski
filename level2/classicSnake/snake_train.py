import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CURRENT_DIR = Path(__file__).resolve().parent

from q_logic_snake import snakeAgent, snakeAgent_head
from q_logic.q_logic_logging import make_run_name, CSVLogger
import random
import time
from q_logic.q_logic import set_seed

set_seed(42)


from environment import SimpleSnakeEnv


def remembrance(count):
    return (0.2+ 0.8*count/200  > random.random())



def main(
    num_games=15000,
    save_every=1000,
    max_steps=500,
    board_size=10,
    advanced_logging=True,
    model_type="modular_dueling_noisy",
    backbone_type="classic",
    priority=True,
    replay_buffer_type=None,
    priority_decay=0.995,
    td_mix=0.5,
    loss_type="huber",
    scheduler_type="warm_restart",
    scheduler_warmup_steps=5_000,
    scheduler_hold_steps=5_000,
    scheduler_decay_steps=500_000,
    scheduler_initial_lr=1e-4,
    scheduler_max_lr=5e-4,
    scheduler_final_lr=1e-6,
):
    polyak = True
    save_dir = CURRENT_DIR / "model_saves"
    log_dir = CURRENT_DIR / "log"
    if replay_buffer_type is None:
        replay_buffer_type = "td" if priority else "uniform"

    for i in [0]:
        for double_q in [True]:
            for noisyNet in [True]:
                gamma = 0.97
                file_name = make_run_name(f"snakeagent1_{backbone_type}_{model_type}_replay{replay_buffer_type}_loss{loss_type}_scheduler{scheduler_type}_polyak{polyak}_gamma{gamma}_doubleq{double_q}_noisynet{noisyNet}zero_survive_reward_ver{i}")

                logger = CSVLogger(str(log_dir / f"{file_name}.csv"), fieldnames=[
                        "game", "avg_count", "avg_reward","avg_jabuka","vrijeme" ])

                agent1 = snakeAgent(gamma= gamma, noisy_net=noisyNet, double_q=double_q, priority = priority,
                                    advanced_logging_path=file_name if advanced_logging else False, polyak = polyak,
                                    save_dir=str(save_dir), model_type=model_type, backbone_type=backbone_type,
                                    replay_buffer_type=replay_buffer_type, priority_decay=priority_decay, td_mix=td_mix,
                                    loss_type=loss_type, scheduler_type=scheduler_type,
                                    scheduler_warmup_steps=scheduler_warmup_steps,
                                    scheduler_hold_steps=scheduler_hold_steps,
                                    scheduler_decay_steps=scheduler_decay_steps,
                                    scheduler_initial_lr=scheduler_initial_lr,
                                    scheduler_max_lr=scheduler_max_lr,
                                    scheduler_final_lr=scheduler_final_lr)
                effective_noisy_net = agent1.noisy_net

                brojac = 0
                avg_count = 10
                avg_reward = 0
                avg_jabuka = 0
                print(f"Starting snake training: games={num_games}, save_every={save_every}, max_steps={max_steps}, replay_buffer={replay_buffer_type}, loss={loss_type}, scheduler={scheduler_type}, warmup={scheduler_warmup_steps}, hold={scheduler_hold_steps}, decay={scheduler_decay_steps}, initial_lr={scheduler_initial_lr}, max_lr={scheduler_max_lr}, final_lr={scheduler_final_lr}, priority_decay={priority_decay}, td_mix={td_mix}, noisy_net={effective_noisy_net}, backbone={backbone_type}, model_type={model_type}", flush=True)
                for game_no in range(num_games):
                    # Create environment with human render mode
                    env = SimpleSnakeEnv(size = board_size)

                    state, snake_state = env.reset()
                    done = False
                    reward = 0
                    count = 0
                    a = time.time()

                    sum_reward = 0
                    jabuka = 0
                    jabuka_novi = 0
                    while (not done and jabuka < 50) :

                        count += 1
                        # Random action just to view the game
                        if effective_noisy_net:
                            action, ratios = agent1.get_action((state,snake_state,reward,jabuka,done))
                        else:
                            action = agent1.get_action((state,snake_state,reward,jabuka,done))

                        state_novi, snake_state_novi,reward_novi, done_novi, info = env.step(action)
                        if count == max_steps:
                            done_novi = True

                        if reward >= 0.5:
                            jabuka_novi += 1

                        if remembrance(count) or True:
                            agent1.remember((state,snake_state,reward,jabuka,done),(state_novi,snake_state_novi,reward_novi,jabuka_novi,done_novi))

                        agent1.train()

                        reward_novi,done_novi = agent1.give_reward((state_novi, snake_state, reward_novi,jabuka_novi, done_novi),(state, snake_state, reward, jabuka,done),action)

                        state = state_novi
                        done = done_novi
                        reward = reward_novi
                        jabuka = jabuka_novi
                        brojac += 1

                        sum_reward += reward
                    if save_every and game_no > 0 and game_no % save_every == 0:
                        agent1.save_agent_state(f"{file_name}_episode_{game_no}")
                        print(f"saved checkpoint: {file_name}_episode_{game_no}.pt", flush=True)

                    avg_count = 0.99*avg_count + 0.01*count
                    avg_reward = 0.99*avg_reward + 0.01*sum_reward/count
                    avg_jabuka = 0.99*avg_jabuka + 0.01*jabuka
                    form = '{:.4f}'
                    if game_no % 20 == 0:
                        print(form.format(game_no), form.format(count),  form.format(avg_count), form.format(sum_reward/count), form.format(avg_reward), form.format(jabuka), form.format(avg_jabuka), f"br treninga = {brojac}", flush=True)
                    vrijeme= time.time()  - a

                    logger.log({
                            "game": game_no,
                            "avg_count": avg_count,
                            "avg_reward": avg_reward,
                            "avg_jabuka": avg_jabuka,
                            "vrijeme": vrijeme
                        })
                agent1.save_agent_state(f"{file_name}_final")
                print(f"saved final model: {file_name}_final.pt", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=15000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--board-size", type=int, default=10)
    parser.add_argument("--no-advanced-logging", action="store_true")
    parser.add_argument("--priority", dest="priority", action="store_true", default=True)
    parser.add_argument("--no-priority", dest="priority", action="store_false")
    parser.add_argument(
        "--replay-buffer",
        choices=["uniform", "td", "td_decay", "td_mix"],
        default=None,
        help="Replay memory type. If omitted, --priority maps to td and --no-priority maps to uniform.",
    )
    parser.add_argument("--priority-decay", type=float, default=0.995)
    parser.add_argument("--td-mix", type=float, default=0.5)
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--scheduler", choices=["warm_restart", "cosine_warmup_hold"], default="warm_restart")
    parser.add_argument("--scheduler-warmup-steps", type=int, default=5_000)
    parser.add_argument("--scheduler-hold-steps", type=int, default=5_000)
    parser.add_argument("--scheduler-decay-steps", type=int, default=500_000)
    parser.add_argument("--scheduler-initial-lr", type=float, default=1e-4)
    parser.add_argument("--scheduler-max-lr", type=float, default=5e-4)
    parser.add_argument("--scheduler-final-lr", type=float, default=1e-6)
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
        num_games=args.num_games,
        save_every=args.save_every,
        max_steps=args.max_steps,
        board_size=args.board_size,
        advanced_logging=not args.no_advanced_logging,
        model_type=args.model_type,
        backbone_type=args.backbone,
        priority=args.priority,
        replay_buffer_type=args.replay_buffer,
        priority_decay=args.priority_decay,
        td_mix=args.td_mix,
        loss_type=args.loss,
        scheduler_type=args.scheduler,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        scheduler_hold_steps=args.scheduler_hold_steps,
        scheduler_decay_steps=args.scheduler_decay_steps,
        scheduler_initial_lr=args.scheduler_initial_lr,
        scheduler_max_lr=args.scheduler_max_lr,
        scheduler_final_lr=args.scheduler_final_lr,
    )
