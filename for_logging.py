import csv
import os

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
