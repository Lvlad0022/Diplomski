from paralell_training_agent import snakeAgent_inference, snakeAgent_trainer
from q_logic.q_logic_logging import make_run_name, CSVLogger
from q_logic.q_logic_memory_classes import TDPriorityReplayBuffer
import os
import random
import time
import sys
from q_logic.q_logic import set_seed

set_seed(42)


from environment import SimpleSnakeEnv
import time


def remembrance(count):
    return (0.2+ 0.8*count/200  > random.random())



def main():
    polyak = True
    for i in [0]:
        for double_q in [True]:
            for priority in [True]:
                for noisyNet in [True]:
                    
                    agent_trainer = snakeAgent_trainer( )
                    agent_explore = snakeAgent_inference( )
                    
                    memory = TDPriorityReplayBuffer(capacity=200000)
                    brojac = 0
                    vrijeme_pocetak = time.time()
                    num_games = 10_000
                    avg_count = 10
                    avg_reward = 0
                    avg_jabuka = 0
                    
                    ukupno_vrijeme = time.time()
                    fetch_time = 0.0
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
                                action, ratios = agent_explore.get_action((state,snake_state,reward,jabuka,done))
                            else:
                                action = agent_explore.get_action((state,snake_state,reward,jabuka,done))
                                
                            state_novi, snake_state_novi,reward_novi, done_novi, info = env.step(action)
                            if count == 500:
                                done_novi = True

                            if reward >= 0.5:
                                jabuka_novi += 1


                           

                            
                            
                            reward_novi,done_novi = agent_explore.give_reward((state_novi, snake_state, reward_novi,jabuka_novi, done_novi),(state, snake_state, reward, jabuka,done),action)
                            
                            exp = agent_explore.remember((state,snake_state,reward,jabuka,done),(state_novi,snake_state_novi,reward_novi,jabuka_novi,done_novi))
                            if isinstance(exp, list) :
                                for m in exp:
                                    memory.push(m)
                            else:
                                memory.push(exp)
                                        
                            

                            state = state_novi
                            done = done_novi
                            reward = reward_novi
                            jabuka = jabuka_novi

                            if len(memory) > 1000:
                                brojac += 1
                                v = time.time()
                                batch = memory.sample(64)
                                fetch_time = time.time() - v
                                samples, data_idxs, weights, sample_priorities, sample_log = batch
                                loss , idxs, td_error, sample_priorities =agent_trainer.train(samples, data_idxs, weights, sample_priorities, sample_log)
                                memory.update_priorities(data_idxs, td_error, sample_priorities)
                                if brojac%10_000 == 0:
                                    print(f"  time={(time.time() - ukupno_vrijeme)*1000:.2f} ")
                                    vrijeme_pocetak = time.time()
                                    fetch_time = 0.0
                                if brojac % 10 == 0:
                                    agent_explore.load_model_state_dict(agent_trainer.get_model_state_dict(), noisynet=noisyNet, training=True)


                            sum_reward += reward
                        
                        avg_count = 0.99*avg_count + 0.01*count
                        avg_reward = 0.99*avg_reward + 0.01*sum_reward/count
                        avg_jabuka = 0.99*avg_jabuka + 0.01*jabuka
                        form = '{:.4f}'
                        if game_no % 20 == 0:
                            print(form.format(game_no), form.format(count),  form.format(avg_count), form.format(sum_reward/count), form.format(avg_reward), form.format(jabuka), form.format(avg_jabuka), f"br treninga = {brojac}") 
                        vrijeme= time.time()  - a
                        
                        

if __name__ == "__main__":
        main()







# nisam jos uspio da explorer, trainer rade dobro