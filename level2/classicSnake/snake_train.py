from q_logic_snake import snakeAgent
from q_logic.q_logic_logging import make_run_name, CSVLogger
import os
import random
import time
import sys
from q_logic.q_logic_univerzalno import set_seed

set_seed(42)


from environment import SimpleSnakeEnv
import time

def main():
    for polyak in [True]:
        for double_q in [True]:
            for priority in [True]:
                for noisyNet in [True]:
                    gamma = 0.99
                    file_name = make_run_name(f"snake__polyak{polyak}_gamma{gamma}_doubleq{double_q}_priority{priority}_noisynet{noisyNet}_no_survive_reward")

                    logger = CSVLogger(file_name, fieldnames=[
                            "game", "avg_count", "avg_reward","avg_jabuka","vrijeme" ])


                    agent1 = snakeAgent(gamma= gamma, noisy_net=noisyNet, double_q=double_q, priority = priority, advanced_logging_path=file_name, polyak = polyak )
                    
                    num_games = 6000
                    avg_count = 10
                    avg_reward = 0
                    avg_jabuka = 0
                    for game_no in range(num_games):
                        if game_no == 350:
                            agent1.reward_policy = True
                        # Create environment with human render mode
                        env = SimpleSnakeEnv(size = 10)

                        state = env.reset()
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
                            if noisyNet:
                                action, ratios = agent1.get_action((state,reward,jabuka,done))
                            else:
                                action = agent1.get_action((state,reward,jabuka,done))
                                
                            state_novi, reward_novi, done_novi, info = env.step(action)
                            if reward >= 0.5:
                                jabuka_novi += 1

                            agent1.remember((state,reward,jabuka,done),(state_novi,reward_novi,jabuka_novi,done_novi))

                            
                            agent1.train()
                            
                            reward,done = agent1.give_reward((state_novi, reward_novi,jabuka_novi, done_novi),(state, reward, jabuka,done),action)
                            
                            state = state_novi
                            done = done_novi
                            reward = reward_novi
                            jabuka = jabuka_novi


                            sum_reward += reward
                            # optional slowdown for visibility
                            #time.sleep(0.01
                        if(game_no%1000 == 0):
                            agent1.save_agent_state(f"{file_name}.pt")
                        
                        avg_count = 0.99*avg_count + 0.01*count
                        avg_reward = 0.99*avg_reward + 0.01*sum_reward/count
                        avg_jabuka = 0.99*avg_jabuka + 0.01*jabuka
                        print(game_no, count,  avg_count, sum_reward/count, avg_reward, jabuka, avg_jabuka) 
                        vrijeme= time.time()  - a
                        logger.log({
                                "game": game_no,
                                "avg_count": avg_count,
                                "avg_reward": avg_reward,
                                "avg_jabuka": avg_jabuka,
                                "vrijeme": vrijeme
                            })

if __name__ == "__main__":
        main()
