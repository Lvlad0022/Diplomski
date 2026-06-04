import unittest

import numpy as np

from level_experiments.noisy_net_circling.q_logic_circling import CirclingAgent


class NoisyNetCirclingAgentTests(unittest.TestCase):
    def test_standard_get_action_and_remember_use_n_step_return(self):
        agent = CirclingAgent(n_step_remember=3, gamma=0.5, noisy_net=False)
        obs0 = np.zeros((2, 20, 20), dtype=np.float32)
        obs1 = np.ones((2, 20, 20), dtype=np.float32)
        obs2 = np.full((2, 20, 20), 2, dtype=np.float32)
        obs3 = np.full((2, 20, 20), 3, dtype=np.float32)

        data0 = (obs0, 0.0, False, {})
        data1 = (obs1, 1.0, False, {})
        data2 = (obs2, 2.0, False, {})
        data3 = (obs3, 4.0, False, {})

        agent.last_action = np.array([0, 1, 0, 0])
        agent.remember(data0, data1)
        self.assertEqual(len(agent.memory), 0)

        agent.last_action = np.array([0, 0, 1, 0])
        agent.remember(data1, data2)
        self.assertEqual(len(agent.memory), 0)

        agent.last_action = np.array([0, 0, 0, 1])
        agent.remember(data2, data3)
        self.assertEqual(len(agent.memory), 1)

        experience = agent.memory.memory[0]
        self.assertEqual(experience.action, 1)
        self.assertAlmostEqual(experience.reward, 1.0 + 0.5 * 2.0 + 0.25 * 4.0)
        self.assertAlmostEqual(experience.gamma, 0.5 ** 3)

        obs4 = np.full((2, 20, 20), 4, dtype=np.float32)
        data4 = (obs4, 8.0, False, {})
        agent.last_action = np.array([1, 0, 0, 0])
        agent.remember(data3, data4)

        self.assertEqual(len(agent.memory), 2)
        experience = agent.memory.memory[1]
        self.assertEqual(experience.action, 2)
        self.assertAlmostEqual(experience.reward, 2.0 + 0.5 * 4.0 + 0.25 * 8.0)
        self.assertAlmostEqual(experience.gamma, 0.5 ** 3)


if __name__ == "__main__":
    unittest.main()
