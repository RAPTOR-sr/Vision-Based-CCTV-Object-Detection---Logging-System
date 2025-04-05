# live_object_captioner/data_logger.py
import csv
import os
from datetime import datetime

class DataLogger:
    """
    Logs detection data (timestamp, category, caption) to a CSV file.
    """
    def __init__(self, log_dir='output', filename='detections.csv'):
        """
        Initializes the logger.

        Args:
            log_dir (str): The directory to store the log file.
            filename (str): The name of the CSV log file.
        """
        self.log_dir = log_dir
        self.filepath = os.path.join(log_dir, filename)
        self.header = ['Timestamp', 'Category', 'Caption']

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Write header if file doesn't exist or is empty
        file_exists = os.path.isfile(self.filepath)
        is_empty = file_exists and os.path.getsize(self.filepath) == 0

        if not file_exists or is_empty:
            try:
                with open(self.filepath, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.header)
                print(f"CSV log file created/header written at: {self.filepath}")
            except IOError as e:
                print(f"Error creating/writing header to CSV file: {e}")
                self.filepath = None # Indicate logging failure

    def log_detection(self, category, caption):
        """
        Appends a detection record to the CSV file.

        Args:
            category (str): The detected object category (class name).
            caption (str): The generated caption for the object.
        """
        if self.filepath is None:
            print("Logging skipped: CSV file path is not set due to earlier error.")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # Timestamp with milliseconds
        row = [timestamp, category, caption]

        try:
            with open(self.filepath, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
        except IOError as e:
            print(f"Error writing to CSV file: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during logging: {e}")