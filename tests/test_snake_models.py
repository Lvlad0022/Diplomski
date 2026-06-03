import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "level2" / "classicSnake"))

from model_snake import build_snake_model


class SnakeModelTests(unittest.TestCase):
    def test_modular_heads_return_q_values(self):
        x = torch.zeros(2, 3, 10, 10)
        for backbone_type in ["classic", "resnext_snake"]:
            for head_type in ["classic", "noisy", "dueling", "dueling_noisy"]:
                with self.subTest(backbone_type=backbone_type, head_type=head_type):
                    model = build_snake_model(backbone_type=backbone_type, head_type=head_type)
                    model.reset_noise()
                    y = model(x)
                    self.assertEqual(tuple(y.shape), (2, 4))

    def test_resnext_snake_backbone_accepts_single_state(self):
        x = torch.zeros(3, 10, 10)
        model = build_snake_model(backbone_type="resnext_snake", head_type="dueling_noisy")
        model.reset_noise()
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 4))

    def test_noisy_heads_support_ratios_output(self):
        x = torch.zeros(2, 3, 10, 10)
        for backbone_type in ["classic", "resnext_snake"]:
            for head_type in ["noisy", "dueling_noisy"]:
                with self.subTest(backbone_type=backbone_type, head_type=head_type):
                    model = build_snake_model(backbone_type=backbone_type, head_type=head_type)
                    model.reset_noise()
                    model.ratios = True
                    q_values, ratios = model(x)
                    self.assertEqual(tuple(q_values.shape), (2, 4))
                    self.assertTrue(ratios)

    def test_invalid_backbone_raises(self):
        with self.assertRaises(ValueError):
            build_snake_model(backbone_type="missing", head_type="classic")

    def test_invalid_head_raises(self):
        with self.assertRaises(ValueError):
            build_snake_model(backbone_type="classic", head_type="missing")


if __name__ == "__main__":
    unittest.main()
