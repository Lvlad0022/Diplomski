import numpy as np
import os
import random
import time
import sys

from q_logic.q_logic import set_seed
from pretraining_snake import snake_pretrainer
from q_logic_snake import snakeAgent, snakeAgent2
from q_logic.q_logic_logging import make_run_name, CSVLogger 

 

set_seed(42)


from environment import SimpleSnakeEnv
import time

def main():

    
    file_name = make_run_name(f"pretrain_model")
    folder_path = "Diplomski/level2/classicSnake/log"
    file_path = f"{folder_path}/{file_name}.csv"
    os.makedirs(folder_path, exist_ok=True)
    logger = CSVLogger(file_path, fieldnames=[
                        "game", "avg_loss" ])


    pretrainer = snake_pretrainer()

    num_games = 6000

    avg_loss = 1
    avg_lossevi = np.ones((pretrainer.num_problems,))

    for game_no in range(num_games):
        env = SimpleSnakeEnv(size = 10)

        state, snake_state = env.reset()
        done = False
        reward = 0
        count = 0
        a = time.time()

        jabuka = 0

        granica = random.randint(5,10)
        sum_loss = 0
        sum_lossevi = np.zeros((pretrainer.num_problems,))
        train_count = 0

        while (not done and jabuka < 50) :
            
            count += 1
            # Random action just to view the game
            action = pretrainer.get_action((state,snake_state,reward,jabuka,done))
                
            state, snake_state,reward, done, info = env.step(action)
            if reward >= 0.5:
                jabuka += 1

            if(count == granica):
                pretrainer.remember((state,snake_state,reward,jabuka,done))
                granica += random.randint(5,10)
                loss, lossevi = pretrainer.train()
                
                if loss: # krece s treinranjem tek kada se napuni buffer do 1000
                    sum_loss += loss
                    sum_lossevi += np.array(lossevi)
                    train_count += 1

        if((game_no+1)%1000 == 0):
            pretrainer.save_agent_state(f"{file_name}")

        if(train_count):
             avg_loss = 0.99* avg_loss + 0.01*(sum_loss/train_count)
             avg_lossevi = 0.99* avg_lossevi + 0.01*(sum_lossevi/train_count)
        
        print(game_no, count, avg_loss)
        print(avg_lossevi) 
        logger.log({
                                "game": game_no,
                                "avg_loss": avg_loss
                            })
        
if __name__ == "__main__":
        main()
