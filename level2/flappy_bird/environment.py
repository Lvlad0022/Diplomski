import numpy as np
import random

class FlappyEnv:
    def __init__(self):
        self.gravity = 0.4
        self.flap_strength = -6
        self.pipe_gap = 100
        self.pipe_speed = 3

        self.width = 288
        self.height = 512

        self.bird_x = 50
        self.reset()

    def reset(self):
        self.bird_y = self.height // 2
        self.bird_vel = 0

        self.pipe_x = 300
        self.pipe_top = random.randint(50, 300)
        self.pipe_bottom = self.pipe_top + self.pipe_gap

        self.score = 0
        self.done = False

        return self._get_state()

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        # Apply action
        if action == 1:
            self.bird_vel = self.flap_strength

        # Gravity
        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel

        # Move pipes
        self.pipe_x -= self.pipe_speed

        # When pipe goes off screen → spawn new
        if self.pipe_x < -50:
            self.pipe_x = 300
            self.pipe_top = random.randint(50, 300)
            self.pipe_bottom = self.pipe_top + self.pipe_gap
            self.score += 1  # passed pipe

        # Check death
        if self.bird_y < 0 or self.bird_y > self.height:
            self.done = True
            return self._get_state(), -1, True, {}

        # Collision with pipe
        if (self.pipe_x < self.bird_x < self.pipe_x + 50):
            if not (self.pipe_top < self.bird_y < self.pipe_bottom):
                self.done = True
                return self._get_state(), -1, True, {}

        # Reward for being alive
        reward = 0.1

        return self._get_state(), reward, False, {}

    def _get_state(self):
        return np.array([
            self.bird_y / self.height,
            self.bird_vel / 10,
            self.pipe_x / self.width,
            self.pipe_top / self.height,
            self.pipe_bottom / self.height,
        ], dtype=np.float32)
