import csv
import os
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter


class CSVLogger:
    def __init__(self, filepath, fieldnames):
        """
        Args:
            filepath (str): Path to the CSV file.
            fieldnames (list of str): Column names for the CSV.
        """
        self.filepath = filepath
        self.fieldnames = fieldnames

        # Create the file and write header if it doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row_dict):
        """Append one row of metrics to the CSV."""
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_dict)


def make_run_name(
    file_name
):
    """
    Automatski generira ime eksperimenta, npr.:
    DQN_dueling_lr0.0005_bs32_seed42_2025-11-11_14-32-05
    """
    parts = [file_name]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parts.append(timestamp)

    return "_".join(parts)


class Advanced_stat_logger:
    def __init__(self, filename, update_every, batch_size, log_dir="logs/advanced_stats"):
        self.filename = make_run_name(filename)
        self.filename_csv = f"{self.filename}.csv"
        self.count = 0
        self.update_every = update_every
        self.batch_size = batch_size
        self.step = 1

        # --- inicijalizacija buffer polja ---
        self.td_vector = np.zeros((update_every * batch_size,))
        self.Q_val_vector = np.zeros((update_every * batch_size,))
        self.episode_count_vector = np.zeros((update_every * batch_size,))

        self.experience_age_vector = np.zeros((update_every * batch_size,))
        self.weights_vector = np.zeros((update_every * batch_size,))
        self.priorities_vector = np.zeros((update_every * batch_size,))
        self.replay_size_vector = np.zeros((update_every,))
        self.norm_vector = np.zeros((update_every,))

        # --- TensorBoard writer ---
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{self.filename}")

        # --- CSV header ---
        self.fieldnames = [
            "td_error25", "td_error50", "td_error90", "td_error99",
            "Q_val25", "Q_val50", "Q_val90", "Q_val99",
            "replay_size",
            "experience_age25", "experience_age50", "experience_age75", "experience_age90",
            "weights50", "weights75", "weights90", "weights99",
            "priorities50", "priorities75", "priorities90", "priorities99",
            "grad_norm25", "grad_norm50", "grad_norm90", "grad_norm99"
        ]

        if not os.path.exists(self.filename_csv):
            with open(self.filename_csv, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    # -----------------------------
    # glavna metoda za dodavanje batch loga
    # -----------------------------
    def __call__(self, train_log, sample_log):
        td_error, Q_val, episode_count, total_norm = train_log
        experience_age, weights, sample_priorities, replay_size = sample_log

        start, end = self.count * self.batch_size, (self.count + 1) * self.batch_size

        self.td_vector[start:end] = td_error
        self.Q_val_vector[start:end] = Q_val
        self.episode_count_vector[start:end] = episode_count
        self.experience_age_vector[start:end] = experience_age
        self.weights_vector[start:end] = weights
        self.priorities_vector[start:end] = sample_priorities

        self.replay_size_vector[self.count] = replay_size
        self.norm_vector[self.count] = total_norm

        self.count += 1
        if self.count == self.update_every:
            self.count = 0
            self.save_log()


    def save_log(self):
        def percentiles(x):
            return {
                "25": np.percentile(x, 25),
                "50": np.percentile(x, 50),
                "75": np.percentile(x, 75),
                "90": np.percentile(x, 90),
                "99": np.percentile(x, 99),
            }

        # --- izračunaj percentilne metrike ---
        td_p = percentiles(self.td_vector)
        q_p = percentiles(self.Q_val_vector)
        age_p = percentiles(self.experience_age_vector)
        w_p = percentiles(self.weights_vector)
        pr_p = percentiles(self.priorities_vector)
        norm_p = percentiles(self.norm_vector)

        replay_size = np.mean(self.replay_size_vector)

        # --- priprema zapisa za CSV ---
        data_row = {
            "td_error25": td_p["25"], "td_error50": td_p["50"], "td_error90": td_p["90"], "td_error99": td_p["99"],
            "Q_val25": q_p["25"], "Q_val50": q_p["50"], "Q_val90": q_p["90"], "Q_val99": q_p["99"],
            "replay_size": replay_size,
            "experience_age25": age_p["25"], "experience_age50": age_p["50"],
            "experience_age75": age_p["75"], "experience_age90": age_p["90"],
            "weights50": w_p["50"], "weights75": w_p["75"], "weights90": w_p["90"], "weights99": w_p["99"],
            "priorities50": pr_p["50"], "priorities75": pr_p["75"],
            "priorities90": pr_p["90"], "priorities99": pr_p["99"],
            "grad_norm25": norm_p["25"], "grad_norm50": norm_p["50"],
            "grad_norm90": norm_p["90"], "grad_norm99": norm_p["99"],
        }

        # --- spremi u CSV ---
        with open(self.filename_csv, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data_row)

        
        step =  self.step # za sad neka je ovako kasnije eventualno nesto informaticvnije

        self.writer.add_scalar("train/replay_size", replay_size, step)
        self.writer.add_scalars("train/td_error", {"p25": td_p["25"], "p50": td_p["50"], "p90": td_p["90"], "p99": td_p["99"]}, step)
        self.writer.add_scalars("train/q_value", {"p25": q_p["25"], "p50": q_p["50"], "p90": q_p["90"], "p99": q_p["99"]}, step)
        self.writer.add_scalars("train/experience_age", {"p25": age_p["25"], "p50": age_p["50"], "p75": age_p["75"], "p90": age_p["90"]}, step)
        self.writer.add_scalars("train/weights", {"p50": w_p["50"], "p75": w_p["75"], "p90": w_p["90"], "p99": w_p["99"]}, step)
        self.writer.add_scalars("train/priorities", {"p50": pr_p["50"], "p75": pr_p["75"], "p90": pr_p["90"], "p99": pr_p["99"]}, step)
        self.writer.add_scalars("train/grad_norm", {"p25": norm_p["25"], "p50": norm_p["50"], "p90": norm_p["90"], "p99": norm_p["99"]}, step)

        # --- reset vektora ---
        self.td_vector.fill(0)
        self.Q_val_vector.fill(0)
        self.episode_count_vector.fill(0)
        self.experience_age_vector.fill(0)
        self.weights_vector.fill(0)
        self.priorities_vector.fill(0)
        self.replay_size_vector.fill(0)
        self.norm_vector.fill(0)
        self.step += 1

    # -----------------------------
    def close(self):
        self.writer.close()



class Time_logger:
    def __init__(self, filename, update_every=1000, log_dir="logs/time_logger"):
        self.filename = make_run_name(filename)
        self.filename_csv = f"{self.filename}.csv"
        self.update_every = update_every
        self.count = 0
        self.step  = 1

        # --- Buffers ---
        self.sample_times = np.zeros((update_every,))
        self.update_priorities_times = np.zeros((update_every,))
        self.logging_times = np.zeros((update_every,))
        self.move_to_gpu_times = np.zeros((update_every,))
        self.forward_times = np.zeros((update_every,))
        self.backprop_times = np.zeros((update_every,))

        # --- TensorBoard ---
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{self.filename}")

        # --- CSV header ---
        self.fieldnames = [
            "sample_mean", "sample_p50", "sample_p90", "sample_p99",
            "update_priorities_mean", "update_priorities_p50", "update_priorities_p90", "update_priorities_p99",
            "logging_mean", "logging_p50", "logging_p90", "logging_p99",
            "move_to_gpu_mean", "move_to_gpu_p50", "move_to_gpu_p90", "move_to_gpu_p99",
            "forward_mean", "forward_p50", "forward_p90", "forward_p99",
            "backprop_mean", "backprop_p50", "backprop_p90", "backprop_p99"
        ]

        if not os.path.exists(self.filename_csv):
            with open(self.filename_csv, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    # ---------------------------------------------------------
    def __call__(self, vremena_long_term, vremena_train):
        """
        Dodaje nove mjerene vrijednosti u buffer.
        Kad se napuni `update_every`, izračunava statistike i logira ih.
        """
        vrijeme_sample, vrijeme_update_priorities, vrijeme_logging = vremena_long_term
        vrijeme_move_to_gpu, vrijeme_forward_prop, vrijeme_back_prop = vremena_train

        idx = self.count
        self.sample_times[idx] = vrijeme_sample
        self.update_priorities_times[idx] = vrijeme_update_priorities
        self.logging_times[idx] = vrijeme_logging
        self.move_to_gpu_times[idx] = vrijeme_move_to_gpu
        self.forward_times[idx] = vrijeme_forward_prop
        self.backprop_times[idx] = vrijeme_back_prop

        self.count += 1

        if self.count == self.update_every:
            self.save_log()
            self.reset_buffers()

    # ---------------------------------------------------------
    def _percentiles(self, arr):
        return {
            "mean": np.mean(arr),
            "p50": np.percentile(arr, 50),
            "p90": np.percentile(arr, 90),
            "p99": np.percentile(arr, 99)
        }

    # ---------------------------------------------------------
    def save_log(self):
        # Izračunaj statistike za svaku kategoriju
        s = self._percentiles(self.sample_times)
        u = self._percentiles(self.update_priorities_times)
        l = self._percentiles(self.logging_times)
        g = self._percentiles(self.move_to_gpu_times)
        f = self._percentiles(self.forward_times)
        b = self._percentiles(self.backprop_times)

        # Priprema reda za CSV
        row = {
            "sample_mean": s["mean"], "sample_p50": s["p50"], "sample_p90": s["p90"], "sample_p99": s["p99"],
            "update_priorities_mean": u["mean"], "update_priorities_p50": u["p50"], "update_priorities_p90": u["p90"], "update_priorities_p99": u["p99"],
            "logging_mean": l["mean"], "logging_p50": l["p50"], "logging_p90": l["p90"], "logging_p99": l["p99"],
            "move_to_gpu_mean": g["mean"], "move_to_gpu_p50": g["p50"], "move_to_gpu_p90": g["p90"], "move_to_gpu_p99": g["p99"],
            "forward_mean": f["mean"], "forward_p50": f["p50"], "forward_p90": f["p90"], "forward_p99": f["p99"],
            "backprop_mean": b["mean"], "backprop_p50": b["p50"], "backprop_p90": b["p90"], "backprop_p99": b["p99"],
        }

        # Zapis u CSV
        with open(self.filename_csv, mode="a", newline="") as f1:
            writer = csv.DictWriter(f1, fieldnames=self.fieldnames)
            writer.writerow(row)

        # TensorBoard log
        step = self.step

        self.writer.add_scalars("time/sample", s, step)
        self.writer.add_scalars("time/update_priorities", u, step)
        self.writer.add_scalars("time/logging", l, step)
        self.writer.add_scalars("time/move_to_gpu", g, step)
        self.writer.add_scalars("time/forward", f, step)
        self.writer.add_scalars("time/backprop", b, step)

        self.step += 1


    # ---------------------------------------------------------
    def reset_buffers(self):
        self.count = 0
        self.sample_times.fill(0)
        self.update_priorities_times.fill(0)
        self.logging_times.fill(0)
        self.move_to_gpu_times.fill(0)
        self.forward_times.fill(0)
        self.backprop_times.fill(0)

    # ---------------------------------------------------------
    def close(self):
        self.writer.close()
