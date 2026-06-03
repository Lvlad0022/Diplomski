import tempfile
import unittest
from pathlib import Path

import numpy as np

from q_logic.q_logic_logging import Advanced_stat_logger, CSVLogger
from q_logic.q_logic_memory_classes import ReplaySampleLog


class LoggingTests(unittest.TestCase):
    def test_csv_logger_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "nested" / "metrics.csv"
            logger = CSVLogger(str(csv_path), fieldnames=["step", "value"])
            logger.log({"step": 1, "value": 2})

            self.assertTrue(csv_path.exists())

    def test_advanced_logger_creates_tensorboard_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = Advanced_stat_logger(
                "logger_smoke",
                update_every=1,
                batch_size=2,
                log_dir=tmpdir,
            )
            train_log = (
                np.array([0.1, 0.2]),
                np.array([1.0, 2.0]),
                np.array([0.0, 0.0]),
                0.5,
                0.123,
            )
            sample_log = ReplaySampleLog(
                np.array([1, 2]),
                np.array([1.0, 1.0]),
                np.array([0.5, 0.7]),
                10,
            )

            logger(train_log, sample_log, step=1, lr=0.001)
            logger.close()

            self.assertTrue(any(Path(tmpdir).rglob("events.out.tfevents.*")))


if __name__ == "__main__":
    unittest.main()
