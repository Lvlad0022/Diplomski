from q_logic_snake import snakeAgent, snakeAgent_metadata
from q_logic.q_logic_logging import make_run_name, CSVLogger
import os
import random
import time
import sys
from q_logic.q_logic import set_seed

set_seed(42)


from environment import SimpleSnakeEnv
import time


def main():
    polyak = True
    for i in [1]:
        for double_q in [True]:
            for priority in [True]:
                for noisyNet in [True]:
                    gamma = 0.99
                    file_name = make_run_name(f"snakeagent1_metadata_dueling_polyak{polyak}_gamma{gamma}_doubleq{double_q}_priority{priority}_noisynet{noisyNet}zero_survive_reward")

                    logger = CSVLogger(file_name, fieldnames=[
                            "game", "avg_count", "avg_reward","avg_jabuka","vrijeme", "lr" ])


                    agent1 = snakeAgent_metadata(gamma= gamma, noisy_net=noisyNet, double_q=double_q, priority = priority, polyak = polyak )
                    
                    num_games = 20_000
                    avg_count = 10
                    avg_reward = 0
                    avg_jabuka = 0
                    for game_no in range(num_games):
                        # Create environment with human render mode
                        env = SimpleSnakeEnv(size = 10)

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
                            if noisyNet:
                                action, ratios = agent1.get_action((state,snake_state,count,reward,jabuka,done))
                            else:
                                action = agent1.get_action((state,snake_state,count,reward,jabuka,done))
                                
                            state_novi, snake_state_novi,reward_novi, done_novi, info = env.step(action)
                            if count == 500:
                                done_novi = True

                            if reward >= 0.5:
                                jabuka_novi += 1

                            agent1.remember((state,snake_state,count,reward,jabuka,done),(state_novi,snake_state_novi,count,reward_novi,jabuka_novi,done_novi))

                            agent1.train(fill=5000)
                            
                            reward_novi,done_novi = agent1.give_reward((state_novi, snake_state, count, reward_novi,jabuka_novi, done_novi),(state, snake_state, count, reward, jabuka,done),action)
                            
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
                        print(game_no, count,  avg_count, sum_reward/count, avg_reward, jabuka, avg_jabuka, agent1.trainer.scheduler.get_lr()) 
                        vrijeme= time.time()  - a
                        logger.log({
                                "game": game_no,
                                "avg_count": avg_count,
                                "avg_reward": avg_reward,
                                "avg_jabuka": avg_jabuka,
                                "vrijeme": vrijeme
                                ,"lr": agent1.trainer.scheduler.get_lr()
                            })

if __name__ == "__main__":
        main()
