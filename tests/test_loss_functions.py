import unittest

import torch

from q_logic.loss_functions import WeightedMSELoss, huberLoss


class LossFunctionTests(unittest.TestCase):
    def test_huber_loss_delta_one_does_not_clip_large_td_error(self):
        loss = huberLoss(delta=1.0)
        pred = torch.tensor([0.0])
        target = torch.tensor([10.0])

        self.assertAlmostEqual(loss(pred, target).item(), 9.5)

    def test_huber_loss_applies_weights(self):
        loss = huberLoss(delta=1.0)
        pred = torch.tensor([0.0, 0.0])
        target = torch.tensor([2.0, 0.5])
        weights = torch.tensor([0.5, 1.0])

        self.assertAlmostEqual(loss(pred, target, weights).item(), 0.4375)

    def test_weighted_mse_loss_matches_unweighted_mse_with_unit_weights(self):
        loss = WeightedMSELoss()
        pred = torch.tensor([1.0, 3.0])
        target = torch.tensor([2.0, 1.0])
        weights = torch.ones(2)

        self.assertAlmostEqual(loss(pred, target, weights).item(), 2.5)


if __name__ == "__main__":
    unittest.main()
