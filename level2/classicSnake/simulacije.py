from environment import SimpleSnakeEnv
import time
import random
from q_logic_snake import snakeAgent

def main():
    # Create environment with human render mode
    env = SimpleSnakeEnv(size = 10)
    agent1 = snakeAgent(train= False)
    
    
    agent1.load_agent_state(r"C:\Users\lovro\Desktop\snake\model_saves\snake__polyakTrue_gamma0.99_doubleqTrue_priorityFalse_2025-11-29_16-37-21.pt")
    

    for i in range(500):
        state = env.reset()
        done = False
        count = 0
        while not done:
            count +=1
            # Random action just to view the game
            if(count < 2):
                action = random.randint(0,3)
            else:
                action = agent1.get_action((state,reward,done))

            state, reward,done, info = env.step(action)
            env.render()
            time.sleep(0.1)

        print(count)
    env.close()

if __name__ == "__main__":
    main()
