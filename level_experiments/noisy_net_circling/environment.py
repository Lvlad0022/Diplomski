import math
import random
from dataclasses import dataclass

import numpy as np


@dataclass
class CirclingState:
    channel: int
    position: tuple[int, int]
    sector: int
    next_sector: int
    sectors_collected: int
    steps: int
    total_steps: int


class CirclingRareChannelEnv:
    """
    Mali environment za testiranje NoisyNet magnitudea u common/rare kanalima.

    Observation je tensor (num_channels, size, size). Samo je aktivni channel
    popunjen mapom: player=1, empty=0, wall=-1. Ostali channeli su nule.
    """

    ACTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }

    def __init__(
        self,
        size=20,
        num_channels=2,
        common_channel_prob=0.95,
        num_sectors=8,
        inner_wall_radius=3.2,
        max_steps=None,
        progress_reward=1.0,
        step_reward=-0.01,
        collision_penalty=-0.05,
        clockwise_shaping_reward=0.0,
        slip_prob=0.0,
        seed=None,
    ):
        if size < 7:
            raise ValueError("size must be at least 7")
        if num_channels < 2:
            raise ValueError("num_channels must be at least 2")
        if not 0.0 <= common_channel_prob <= 1.0:
            raise ValueError("common_channel_prob must be between 0 and 1")

        self.size = size
        self.num_channels = num_channels
        self.common_channel_prob = common_channel_prob
        self.num_sectors = num_sectors
        self.inner_wall_radius = inner_wall_radius
        self.max_steps = max_steps
        self.progress_reward = progress_reward
        self.step_reward = step_reward
        self.collision_penalty = collision_penalty
        self.clockwise_shaping_reward = clockwise_shaping_reward
        self.slip_prob = slip_prob

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.center = ((size - 1) / 2.0, (size - 1) / 2.0)
        self.wall_mask = self._build_wall_mask()
        self.channel_probs = self._build_channel_probs()

        self.action_space_n = 4
        self.reset()

    def reset(self, channel=None):
        self.total_steps = 0
        return self._reset_position(channel=channel, reset_event=True)

    def _reset_position(self, channel=None, reset_event=False):
        self.channel = self._sample_channel() if channel is None else int(channel)
        if not 0 <= self.channel < self.num_channels:
            raise ValueError(f"channel must be in [0, {self.num_channels - 1}]")

        self.position = self._start_position()
        self.steps = 0
        self.done = False

        current_sector = self._sector(self.position)
        self.start_sector = current_sector
        self.next_sector = (current_sector + 1) % self.num_sectors
        self.sectors_collected = 0
        self.visited_target_sectors = set()
        self.last_lap_completed = False
        self.last_reset_event = reset_event

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        if action not in self.ACTIONS:
            raise ValueError(f"Unsupported action: {action}")

        if self.slip_prob > 0 and self.rng.random() < self.slip_prob:
            action = self.rng.choice(list(self.ACTIONS.keys()))

        reward = self.step_reward
        dr, dc = self.ACTIONS[action]
        row, col = self.position
        candidate = (row + dr, col + dc)

        if self._is_blocked(candidate):
            reward += self.collision_penalty
        else:
            reward += self._clockwise_shaping(dr, dc)
            self.position = candidate
            reward += self._progress_reward()

        self.steps += 1
        self.total_steps += 1
        lap_completed = self.sectors_collected >= self.num_sectors
        self.last_lap_completed = lap_completed

        if lap_completed or (self.max_steps is not None and self.steps >= self.max_steps):
            obs, info = self._reset_position(channel=None, reset_event=True)
            info["lap_completed"] = lap_completed
            return obs, reward, False, info

        self.last_reset_event = False
        return self._get_obs(), reward, False, self._get_info()

    def render(self):
        print(self.render_text())

    def render_text(self):
        wall_symbol = str((self.channel % 9) + 1)
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        for row in range(self.size):
            for col in range(self.size):
                if self.wall_mask[row, col]:
                    grid[row][col] = wall_symbol

        row, col = self.position
        grid[row][col] = "M"

        lines = [
            f"channel={self.channel} sector={self._sector(self.position)} "
            f"next={self.next_sector} collected={self.sectors_collected}/{self.num_sectors} "
            f"steps={self.steps} total_steps={self.total_steps}",
        ]
        lines.extend(" ".join(row_values) for row_values in grid)
        return "\n".join(lines)

    def _get_obs(self):
        obs = np.zeros((self.num_channels, self.size, self.size), dtype=np.float32)
        channel_map = obs[self.channel]
        channel_map[self.wall_mask] = -1.0
        row, col = self.position
        channel_map[row, col] = 1.0
        return obs

    def _get_info(self):
        return {
            "state": CirclingState(
                channel=self.channel,
                position=self.position,
                sector=self._sector(self.position),
                next_sector=self.next_sector,
                sectors_collected=self.sectors_collected,
                steps=self.steps,
                total_steps=self.total_steps,
            ),
            "channel": self.channel,
            "position": self.position,
            "sector": self._sector(self.position),
            "next_sector": self.next_sector,
            "sectors_collected": self.sectors_collected,
            "steps": self.steps,
            "total_steps": self.total_steps,
            "channel_probability": self.channel_probs[self.channel],
            "lap_completed": self.last_lap_completed,
            "reset_event": self.last_reset_event,
        }

    def _progress_reward(self):
        current_sector = self._sector(self.position)
        if current_sector != self.next_sector:
            return 0.0

        self.visited_target_sectors.add(current_sector)
        self.sectors_collected += 1
        self.next_sector = (self.next_sector + 1) % self.num_sectors
        return self.progress_reward

    def _clockwise_shaping(self, dr, dc):
        if self.clockwise_shaping_reward <= 0:
            return 0.0

        row, col = self.position
        center_row, center_col = self.center
        dy = row - center_row
        dx = col - center_col

        tangent_row = dx
        tangent_col = -dy
        norm = math.hypot(tangent_row, tangent_col)
        if norm == 0:
            return 0.0

        alignment = (dr * tangent_row + dc * tangent_col) / norm
        return self.clockwise_shaping_reward * max(0.0, alignment)

    def _build_channel_probs(self):
        if np.isclose(self.common_channel_prob, 1.0 / self.num_channels):
            probs = np.full(self.num_channels, 1.0 / self.num_channels, dtype=np.float64)
            return probs

        rare_prob = (1.0 - self.common_channel_prob) / (self.num_channels - 1)
        probs = np.full(self.num_channels, rare_prob, dtype=np.float64)
        probs[0] = self.common_channel_prob
        return probs

    def _sample_channel(self):
        return int(self.np_rng.choice(self.num_channels, p=self.channel_probs))

    def _build_wall_mask(self):
        wall_mask = np.zeros((self.size, self.size), dtype=bool)
        center_row, center_col = self.center

        for row in range(self.size):
            for col in range(self.size):
                border = row == 0 or col == 0 or row == self.size - 1 or col == self.size - 1
                dist = math.hypot(row - center_row, col - center_col)
                inner_wall = dist <= self.inner_wall_radius
                wall_mask[row, col] = border or inner_wall

        return wall_mask

    def _start_position(self):
        preferred = (1, self.size // 2)
        if not self._is_blocked(preferred):
            return preferred

        for row in range(self.size):
            for col in range(self.size):
                if not self._is_blocked((row, col)):
                    return (row, col)
        raise RuntimeError("No free cell available")

    def _is_blocked(self, position):
        row, col = position
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return True
        return bool(self.wall_mask[row, col])

    def _sector(self, position):
        row, col = position
        center_row, center_col = self.center
        angle = math.atan2(row - center_row, col - center_col)
        if angle < 0:
            angle += 2 * math.pi
        sector_size = 2 * math.pi / self.num_sectors
        return int(angle // sector_size) % self.num_sectors
