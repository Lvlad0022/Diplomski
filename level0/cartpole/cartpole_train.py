import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from q_logic_cartpole import catrpoleAgent
from q_logic.q_logic_logging import make_run_name, CSVLogger
import argparse
import time
from q_logic.q_logic import set_seed

set_seed(42)


import gymnasium as gym


def main(num_games=6000, save_every=1000, advanced_logging=True):
    print("Starting training...", flush=True)
    for polyak in [True, False]:
        for double_q in [ True]:
            for priority in [False, True]:
                gamma = 0.99

                file_name = make_run_name(f"carptpole__polyak{polyak}_gamma{gamma}_doubleq{double_q}_priority{priority}")
                advanced_logging_path = file_name if advanced_logging else False

    #            logger = CSVLogger(file_name, fieldnames=[
    #                    "game", "avg_count", "vrijeme" ])

                print(f"Starting training with polyak={polyak}, double_q={double_q}, priority={priority}", flush=True)
                agent1 = catrpoleAgent(gamma= gamma, double_q=double_q, priority = priority, advanced_logging_path=advanced_logging_path, polyak = polyak )
                avg_count = 10
                for game_no in range(num_games):
                    
                    # Create environment with human render mode
                    env = gym.make("CartPole-v1")

                    state, info = env.reset()
                    done = False
                    count = 0
                    a = time.time()
                    while not done:
                        count += 1
                        # Random action just to view the game
                        action = agent1.get_action((state,done))

                        state_novi, reward, terminated, truncated, info = env.step(action)
                        done_novi = terminated or truncated

                        agent1.remember((state,done),(state_novi,done_novi))

                        
                        agent1.train()

                        state = state_novi
                        done = done_novi

                        # optional slowdown for visibility
                        #time.sleep(0.01
                    avg_count = 0.99*avg_count + 0.01*count
                    print(game_no, count, avg_count, flush=True) 
                    if save_every and game_no > 0 and game_no % save_every == 0:
                        agent1.save_agent_state(f"{file_name}_episode_{game_no}")
                        print(f"saved checkpoint: {file_name}_episode_{game_no}.pt", flush=True)
                    vrijeme= time.time()  - a
     
    #                logger.log({
    #                        "game": game_no,
    #                        "avg_count": avg_count,
    #                        "vrijeme": vrijeme
    #                    })
                    env.close()
                agent1.save_agent_state(f"{file_name}_final")
                print(f"saved final model: {file_name}_final.pt", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=6000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--no-advanced-logging", action="store_true")
    args = parser.parse_args()
    main(
        num_games=args.num_games,
        save_every=args.save_every,
        advanced_logging=not args.no_advanced_logging,
    )
