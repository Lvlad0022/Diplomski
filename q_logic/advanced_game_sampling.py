import random 
import numpy as np

"""
env sampler that will sample
game configurations out of already played games so further into training player will interact more with complicated positions
level 0 is start of the game 
and level go on 
"""

class env_sampler:
    def __init__(self, num_levels = 2, level_end_probability = [0.4,0.6], level_start_probability = [1,0], aneal_steps = 2000, level_function = None):
        self.capacity = 10_000
        self.memory = [[]for i in range(num_levels-1)]
        self.count = [0 for i in range(num_levels-1)]
        self.done
        self.num_levels = num_levels
        
        self.start_probability = np.array(level_start_probability)
        self.end_probability = np.array(level_end_probability)
        self.probability = self.start_probability
        self.step_counter = 0
        self.aneal_steps = aneal_steps

        self.level_funciton = level_function


    def add_env(self, game, log = None):
        if self.level_function:
            level = self.level_funciton(game)
            if level == 0:
                return
            level -= 1
            if(self.count[level] < self.capacity):
                self.memory[level].append((game,log))
            else:
                self.memory[self.count[level]%self.capacity] = (game,log)
            self.count[level] += 1

    def get_env(self, new_game):
        choice = random.choices(population = range(self.num_levels), weights= self.probability)[0]
        self.update_probability()
        if choice == 0:
            return new_game
        if len(self.memory[choice-1]) == 0:
            return new_game 
        return random.choice(self.memory[choice-1])


    def update_probability(self):
        self.step_counter += 1
        self.step_counter = max(self.step_counter, self.aneal_steps)
        self.probability = self.start_probability + self.step_counter/self.aneal_steps * (self.end_probability-self.start_probability)
    
