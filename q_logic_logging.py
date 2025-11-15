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
        

        # --- inicijalizacija buffer polja ---
        self.td_vector = np.zeros((update_every * batch_size,))
        self.Q_val_vector = np.zeros((update_every * batch_size,))
        self.episode_count_vector = np.zeros((update_every * batch_size,))

        self.experience_age_vector = np.zeros((update_every * batch_size,))
        self.weights_vector = np.zeros((update_every * batch_size,))
        self.priorities_vector = np.zeros((update_every * batch_size,))
        self.replay_size_vector = np.zeros((update_every,))
        self.norm_vector = np.zeros((update_every,))
        self.loss = np.zeros((update_every,))
        self.lr_vector = np.zeros((update_every,))

        # --- TensorBoard writer ---
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{self.filename}")



    
    def __call__(self, train_log, sample_log, step, lr):
        td_error, Q_val, episode_count, total_norm, loss = train_log
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
        self.loss[self.count] = loss
        self.lr_vector[self.count] = lr

        self.count += 1
        if self.count == self.update_every:
            self.count = 0
            self.save_log(step)


    def save_log(self, step):
        def percentiles(x):
            return {
                "1": np.percentile(x, 1),
                "10": np.percentile(x, 10),
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
        loss_mean ={"mean": np.mean(self.loss)}
        lr_mean = {"mean": np.mean(self.lr_vector)}
        
        
        self.writer.add_scalars("train/td_error", td_p, step)
        self.writer.add_scalars("train/q_value", q_p, step)
        self.writer.add_scalars("train/experience_age", age_p, step)
        self.writer.add_scalars("train/weights", w_p, step)
        self.writer.add_scalars("train/priorities", pr_p, step)
        self.writer.add_scalars("train/grad_norm", norm_p, step)
        self.writer.add_scalars("train/loss", loss_mean, step)
        self.writer.add_scalars("train/loss", loss_mean, step)
        
        # --- reset vektora ---
        self.td_vector.fill(0)
        self.Q_val_vector.fill(0)
        self.episode_count_vector.fill(0)
        self.experience_age_vector.fill(0)
        self.weights_vector.fill(0)
        self.priorities_vector.fill(0)
        self.replay_size_vector.fill(0)
        self.norm_vector.fill(0)
        self.loss.fill(0)
        

    # -----------------------------
    def close(self):
        self.writer.close()



class Time_logger:
    def __init__(self, filename, update_every=1000, log_dir="logs/time_logger"):
        self.filename = make_run_name(filename)
        self.filename_csv = f"{self.filename}.csv"
        self.update_every = update_every
        self.count = 0

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
            # sample times
            "sample_mean", "sample_p25", "sample_p50", "sample_p75", "sample_p90", "sample_p99",

            # update priorities times
            "update_priorities_mean", "update_priorities_p25", "update_priorities_p50",
            "update_priorities_p75", "update_priorities_p90", "update_priorities_p99",

            # logging times
            "logging_mean", "logging_p25", "logging_p50", "logging_p75",
            "logging_p90", "logging_p99",

            # move to gpu times
            "move_to_gpu_mean", "move_to_gpu_p25", "move_to_gpu_p50",
            "move_to_gpu_p75", "move_to_gpu_p90", "move_to_gpu_p99",

            # forward times
            "forward_mean", "forward_p25", "forward_p50", "forward_p75",
            "forward_p90", "forward_p99",

            # backprop times
            "backprop_mean", "backprop_p25", "backprop_p50", "backprop_p75",
            "backprop_p90", "backprop_p99",
        ]


        if not os.path.exists(self.filename_csv):
            with open(self.filename_csv, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    # ---------------------------------------------------------
    def __call__(self, vremena_long_term, vremena_train,step):
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
            self.save_log(step)
            self.reset_buffers()

    # ---------------------------------------------------------
    

    # ---------------------------------------------------------
    def save_log(self,step):
        def percentiles(x):
            return {
                "mean": np.mean(x),
                "p25": np.percentile(x, 25),
                "p50": np.percentile(x, 50),
                "p75": np.percentile(x, 75),
                "p90": np.percentile(x, 90),
                "p99": np.percentile(x, 99),
            }
        
        # Izračunaj statistike za svaku kategoriju
        sample_p = percentiles(self.sample_times)
        update_p = percentiles(self.update_priorities_times)
        logging_p = percentiles(self.logging_times)
        gpu_p = percentiles(self.move_to_gpu_times)
        forward_p = percentiles(self.forward_times)
        backprop_p = percentiles(self.backprop_times)


        self.writer.add_scalars("time/sample", sample_p, step)
        self.writer.add_scalars("time/update_priorities", update_p, step)
        self.writer.add_scalars("time/logging", logging_p, step)
        self.writer.add_scalars("time/move_to_gpu", gpu_p, step)
        self.writer.add_scalars("time/forward", forward_p, step)
        self.writer.add_scalars("time/backprop", backprop_p, step)



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
