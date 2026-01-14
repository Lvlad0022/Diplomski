from environment import SimpleSnakeEnv
import time
import random
from q_logic_snake import snakeAgent
from pathlib import Path


def main():
    # Create environment with human render mode
    env = SimpleSnakeEnv(size = 10)
    agent1 = snakeAgent(train= True, noisy_net= True)
    
    current_dir = Path(__file__).parent
    file_name = "snakeagent1__polyakTrue_gamma0.99_doubleqTrue_priorityTrue_noisynetTruezero_survive_reward_ver0_2026-01-13_13-12-24.pt.pt"
    model_path = current_dir/ "representative_models" / file_name
    agent1.load_agent_state(model_path, noisynet=True, training=False)
    
    
    sum_jabuke = 0
    for i in range(100):
        state, snake = env.reset()
        done = False
        count = 0
        jabuka = 0
        reward = 0
        jabuke = 0
        while not done:
            count +=1
            # Random action just to view the game
            if(count <0):
                action = random.randint(0,3)
            else:
                action = agent1.get_action((state,snake,reward,jabuka,done))

            if reward >= 0.5:
                sum_jabuke += 1
                jabuke += 1
            state, snake, reward,done, info = env.step(action)
            env.render()
            time.sleep(0.03)
        print(jabuke)
        time.sleep(1)

if __name__ == "__main__":
    main()
