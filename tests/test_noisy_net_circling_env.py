import unittest

import numpy as np

from level_experiments.noisy_net_circling import CirclingRareChannelEnv


class NoisyNetCirclingEnvTests(unittest.TestCase):
    def test_observation_has_only_active_channel_filled(self):
        env = CirclingRareChannelEnv(seed=1)
        obs, info = env.reset(channel=1)

        self.assertEqual(obs.shape, (2, 20, 20))
        self.assertEqual(info["channel"], 1)
        self.assertTrue(np.any(obs[1] != 0))
        for channel in range(2):
            if channel != 1:
                self.assertTrue(np.all(obs[channel] == 0))

    def test_channel_probabilities_match_common_rare_setup_by_default(self):
        env = CirclingRareChannelEnv()

        np.testing.assert_allclose(env.channel_probs, np.array([0.95, 0.05]))
        self.assertAlmostEqual(float(env.channel_probs.sum()), 1.0)

    def test_clockwise_sector_progress_gets_reward_once_per_expected_sector(self):
        env = CirclingRareChannelEnv(size=10, inner_wall_radius=1.6, seed=1, step_reward=0.0, collision_penalty=0.0)
        env.reset(channel=0)

        # From default start, moving right crosses from sector 6 to expected sector 7.
        _, reward, done, info = env.step(1)
        self.assertEqual(reward, 0.0)

        _, reward, done, info = env.step(1)
        self.assertEqual(reward, 0.0)

        _, reward, done, info = env.step(1)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["sectors_collected"], 1)
        self.assertEqual(info["next_sector"], 0)

    def test_full_lap_resets_without_terminal_done(self):
        env = CirclingRareChannelEnv(size=10, inner_wall_radius=1.6, seed=1, step_reward=0.0, collision_penalty=0.0)
        obs, info = env.reset(channel=0)

        done = False
        for _ in range(80):
            row, col = env.position
            center_row, center_col = env.center
            dy = row - center_row
            dx = col - center_col
            tangent_row = dx
            tangent_col = -dy
            scored_actions = []
            for action, (dr, dc) in env.ACTIONS.items():
                if not env._is_blocked((row + dr, col + dc)):
                    scored_actions.append((dr * tangent_row + dc * tangent_col, action))
            action = sorted(scored_actions, reverse=True)[0][1]
            obs, reward, done, info = env.step(action)
            if info["lap_completed"]:
                break

        self.assertFalse(done)
        self.assertTrue(info["lap_completed"])
        self.assertTrue(info["reset_event"])
        self.assertEqual(info["sectors_collected"], 0)

    def test_clockwise_shaping_rewards_tangent_movement(self):
        env = CirclingRareChannelEnv(
            size=10,
            inner_wall_radius=1.6,
            seed=1,
            step_reward=0.0,
            collision_penalty=0.0,
            progress_reward=0.0,
            clockwise_shaping_reward=0.1,
        )
        env.reset(channel=0)

        _, reward, _, _ = env.step(1)

        self.assertGreater(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
