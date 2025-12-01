from environment import SimpleSnakeEnv
import time
import random
from q_logic_snake import snakeAgent

def main():
    # Create environment with human render mode
    env = SimpleSnakeEnv(size = 10)
    agent1 = snakeAgent(train= False, noisy_net= True)
    
    
    agent1.load_agent_state(r"C:\Users\lovro\Desktop\snake\model_saves\snake__polyakTrue_gamma0.99_doubleqTrue_priorityTrue_noisynetTrue_no_survive_reward_2025-12-01_14-33-12.pt.pt", noisynet=True)
    

    for i in range(500):
        state = env.reset()
        done = False
        count = 0
        jabuka = 0
        reward = 0
        while not done:
            count +=1
            # Random action just to view the game
            if(count <0):
                action = random.randint(0,3)
            else:
                action, _ = agent1.get_action((state,reward,jabuka,done))

            state, reward,done, info = env.step(action)
            env.render()
            time.sleep(0.1)

        print(count)
    env.close()

if __name__ == "__main__":
    main()
