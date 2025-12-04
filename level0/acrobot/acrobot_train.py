from q_logic_acrobot import acrobot_agent
from q_logic.q_logic_logging import make_run_name, CSVLogger
import os
import random
import time
import sys
from Diplomski.q_logic.q_logic import set_seed

set_seed(42)


import gymnasium as gym
import time

def main():
    for polyak in [True]:
        for double_q in [ True, False]:
            for priority in [False, True]:
                gamma = 0.99
                file_name = make_run_name(f"acrobot__polyak{polyak}_gamma{gamma}_doubleq{double_q}_priority{priority}")

                logger = CSVLogger(file_name, fieldnames=[
                        "game", "avg_count", "avg_reward","vrijeme" ])


                agent1 = acrobot_agent(gamma= gamma, double_q=double_q, priority = priority, advanced_logging_path=file_name, polyak = polyak )
                num_games = 6000
                avg_count = 10
                avg_reward = 0
                for game_no in range(num_games):
                    
                    # Create environment with human render mode
                    env = gym.make("Acrobot-v1")

                    state, info = env.reset()
                    done = False
                    count = 0
                    a = time.time()

                    sum_reward = 0
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

                        reward,done = agent1.give_reward((state_novi,done_novi),(state, done),action)

                        sum_reward += reward
                         # optional slowdown for visibility
                        #time.sleep(0.01
                    if(game_no%1000 == 0):
                        agent1.save_agent_state(f"{file_name}.pt")
                    
                    avg_count = 0.99*avg_count + 0.01*count
                    avg_reward = 0.99*avg_reward + 0.01*sum_reward/count
                    print(game_no, count,  avg_count, sum_reward/count,avg_reward) 
                    vrijeme= time.time()  - a
                    logger.log({
                            "game": game_no,
                            "avg_count": avg_count,
                            "avg_reward": avg_reward,
                            "vrijeme": vrijeme
                        })
                    env.close()

if __name__ == "__main__":
        main()
