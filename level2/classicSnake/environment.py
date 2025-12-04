import numpy as np
import random

class SimpleSnakeEnv:
    def __init__(self, size=10, apple_reward=1.0, step_reward=0.1, death_penalty=-1.0):
        self.size = size
        self.apple_reward = apple_reward
        self.step_reward = step_reward
        self.death_penalty = death_penalty
        self.action_space_n = 4   # up, right, down, left

        self.reset()

    # ---------------------
    # Reset
    # ---------------------
    def reset(self):
        c = self.size // 2

        # Snake starts of length 3
        self.snake = [(c, c), (c, c - 1), (c, c - 2)]
        self.direction = (0, 1)  # moving right
        self.done = False
        self.steps = 0

        self._spawn_apple()

        return self._get_obs(), self.snake

    # ---------------------
    # Step
    # ---------------------
    def step(self, action):
        if self.done:
            return self._get_obs(),self.snake, 0.0, True, {}

        # Convert action to direction
        if action == 0:
            new_dir = (-1, 0)  # up
        elif action == 1:
            new_dir = (0, 1)   # right
        elif action == 2:
            new_dir = (1, 0)   # down
        else:
            new_dir = (0, -1)  # left

        # Prevent reversing
        if (new_dir[0] == -self.direction[0] and new_dir[1] == -self.direction[1]):
            new_dir = self.direction

        self.direction = new_dir

        head_x, head_y = self.snake[0]
        nx = head_x + self.direction[0]
        ny = head_y + self.direction[1]

        # Collision: wall
        if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
            self.done = True
            return self._get_obs(),self.snake, self.death_penalty, True, {}

        # Collision: self
        if (nx, ny) in self.snake:
            self.done = True
            return self._get_obs(),self.snake, self.death_penalty, True, {}

        # Move snake
        new_head = (nx, ny)
        self.snake.insert(0, new_head)

        reward = self.step_reward  # reward for surviving

        # Check apple
        if new_head == self.apple:
            reward += self.apple_reward
            # grow snake → do NOT pop tail
            self._spawn_apple()
        else:
            # normal movement → remove tail
            self.snake.pop()

        self.steps += 1
        return self._get_obs(),self.snake,  reward, False, {}

    # ---------------------
    # Apple spawn
    # ---------------------
    def _spawn_apple(self):
        empty_cells = [(i, j)
                       for i in range(self.size)
                       for j in range(self.size)
                       if (i, j) not in self.snake]

        self.apple = random.choice(empty_cells)

    # ---------------------
    # Observation
    # ---------------------
    def _get_obs(self):
        # Channels:
        # 0: snake head
        # 1: snake body
        # 2: apple
        obs = np.zeros((3, self.size, self.size), dtype=np.float32)

        # Head
        hx, hy = self.snake[0]
        obs[0, hx, hy] = 1.0

        # Body
        for (x, y) in self.snake[1:]:
            obs[1, x, y] = 1.0

        # Apple
        ax, ay = self.apple
        obs[2, ax, ay] = 1.0

        return obs

    # ---------------------
    # Render (ASCII)
    # ---------------------
    def render(self):
        grid = np.full((self.size, self.size), '.', dtype=str)

        # Body
        for (x, y) in self.snake:
            grid[x, y] = 'o'

        # Head
        hx, hy = self.snake[0]
        grid[hx, hy] = 'H'

        # Apple
        ax, ay = self.apple
        grid[ax, ay] = 'A'

        print("\n".join(" ".join(row) for row in grid))
        print()
