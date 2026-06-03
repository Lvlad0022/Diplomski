import unittest

import numpy as np

from q_logic.q_logic_memory_classes import (
    Experience,
    ReplayBuffer,
    ReplaySampleLog,
    RewardPriorityReplayBuffer,
    SumTree,
    TDDecayPriorityReplayBuffer,
    TDMixPriorityReplayBuffer,
    TDPriorityReplayBuffer,
)


def make_experience(state, reward=1.0):
    return Experience(
        state=state,
        action=0,
        reward=reward,
        next_state=state + 1,
        done=False,
        gamma=0.99,
    )


class ReplayBufferTests(unittest.TestCase):
    def test_replay_buffer_overwrites_oldest_experience(self):
        memory = ReplayBuffer(capacity=2)

        memory.push(make_experience(1))
        memory.push(make_experience(2))
        memory.push(make_experience(3))

        self.assertEqual(len(memory), 2)
        self.assertEqual(memory.counter, 1)
        self.assertEqual({experience.state for experience in memory.memory}, {2, 3})

    def test_uniform_sample_returns_named_sample_log(self):
        memory = ReplayBuffer(capacity=4)
        for state in range(4):
            memory.push(make_experience(state))

        samples, data_idxs, weights, priorities, sample_log = memory.sample(3)

        self.assertEqual(len(samples), 3)
        self.assertEqual(len(data_idxs), 3)
        np.testing.assert_array_equal(weights, np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(priorities, np.ones(3, dtype=np.float32))
        self.assertIsInstance(sample_log, ReplaySampleLog)
        self.assertEqual(sample_log.replay_size, 4)

    def test_reward_priority_uses_experience_reward_field(self):
        memory = RewardPriorityReplayBuffer(
            capacity=4,
            reward_priority=2,
            alpha=1.0,
            alpha_end=1.0,
            weights=False,
        )

        memory.push(make_experience(1, reward=-3.0))

        self.assertAlmostEqual(memory.priorities.total, 7.0)

    def test_priority_sample_returns_raw_old_priorities(self):
        memory = TDPriorityReplayBuffer(
            capacity=4,
            alpha_start=0.5,
            alpha_end=0.5,
            weights=False,
            priority_clip=10.0,
        )
        for state in range(4):
            memory.push(make_experience(state))

        _, _, _, old_priorities, sample_log = memory.sample(2)

        np.testing.assert_allclose(old_priorities, np.full(2, 5.0, dtype=np.float32), rtol=1e-5)
        np.testing.assert_allclose(sample_log.priorities, old_priorities, rtol=1e-5)

    def test_base_priority_policy_keeps_old_priorities(self):
        memory = ReplayBuffer(capacity=4)
        old_priorities = np.array([2.0, 3.0], dtype=np.float32)

        np.testing.assert_array_equal(memory.priority_policy(np.array([9.0, 9.0]), old_priorities), old_priorities)

    def test_td_priority_policy_uses_clipped_td_error(self):
        memory = TDPriorityReplayBuffer(capacity=4, weights=False, priority_clip=2.0, eps=1e-6)
        memory.push(make_experience(1))

        memory.update_priorities([0], np.array([10.0], dtype=np.float32), np.array([5.0], dtype=np.float32))

        leaf_idx = memory.priorities.leaf_index_from_data_idx(0)
        self.assertAlmostEqual(memory.priorities.tree[leaf_idx], 2.0 ** memory.alpha)

    def test_td_decay_priority_policy_keeps_max_of_td_and_decayed_old_priority(self):
        memory = TDDecayPriorityReplayBuffer(
            capacity=4,
            weights=False,
            priority_decay=0.9,
            eps=0.0,
        )
        td = np.array([1.0, 8.0], dtype=np.float32)
        old_priorities = np.array([10.0, 5.0], dtype=np.float32)

        priorities = memory.priority_policy(td, old_priorities)

        np.testing.assert_allclose(priorities, np.array([9.0, 8.0], dtype=np.float32))

    def test_td_mix_priority_policy_blends_td_and_old_priority(self):
        memory = TDMixPriorityReplayBuffer(
            capacity=4,
            weights=False,
            td_mix=0.25,
            eps=0.0,
        )
        td = np.array([2.0, 10.0], dtype=np.float32)
        old_priorities = np.array([6.0, 2.0], dtype=np.float32)

        priorities = memory.priority_policy(td, old_priorities)

        np.testing.assert_allclose(priorities, np.array([5.0, 4.0], dtype=np.float32))

    def test_sum_tree_segment_sample_returns_named_sample_log(self):
        tree = SumTree(capacity=4)
        for idx, priority in enumerate([1.0, 2.0, 3.0, 4.0]):
            tree.add(priority, idx)

        data_idxs, weights, priorities, sample_log = tree.sample_segment(2)

        self.assertEqual(len(data_idxs), 2)
        self.assertEqual(len(weights), 2)
        self.assertEqual(len(priorities), 2)
        self.assertIsInstance(sample_log, ReplaySampleLog)
        self.assertEqual(sample_log.replay_size, 4)


if __name__ == "__main__":
    unittest.main()
