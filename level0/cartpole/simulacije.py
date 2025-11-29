import gymnasium as gym
import time
import random
from q_logic_cartpole import catrpoleAgent


def main():
    # Create environment with human render mode
    env = gym.make("CartPole-v1",render_mode="human")
    agent1 = catrpoleAgent(train= False)
    
    
    agent1.load_agent_state(r"C:\Users\lovro\Desktop\snake\zzzz2_save.pt")
    

    for i in range(500):
        state, info = env.reset()
        done = False
        count = 0
        while not done:
            count +=1
            # Random action just to view the game
            if(count < -1):
                action = random.randint(0,0)
            else:
                action = agent1.get_action((state,done))

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # optional slowdown for visibility
            time.sleep(0.01)
        print(count)
    env.close()

if __name__ == "__main__":
    main()
