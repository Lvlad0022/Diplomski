import gymnasium as gym
import time
import random
from q_logic_acrobot import acrobot_agent


def main():
    # Create environment with human render mode
    env = gym.make("Acrobot-v1",render_mode="human")
    agent1 = acrobot_agent(train= False)
    
    
    agent1.load_agent_state(r"C:\Users\lovro\Desktop\snake\model_saves\acrobot__polyakTrue_gamma0.99_doubleqTrue_priorityFalse_2025-11-28_23-40-15.pt.pt")
    

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
