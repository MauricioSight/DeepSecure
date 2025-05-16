import os

import pandas as pd
import numpy as np

class SlidingWindowGenerator:
    def __init__(self, config, logger):
        self.logger = logger

        self.phase = config.get("phase")
        self.dataset_config = config.get('dataset')
        self.raw_y_path = self.dataset_config.get('raw_y_path')
        self.raw_x_path = self.dataset_config.get('raw_x_path')
        self.processed_path = self.dataset_config.get("processed_path")

        self.feature_config = config.get('feature')
        self.feature_generator = self.feature_config.get('generator')
        self.window_size = self.feature_config.get('window_size')
        self.window_stride = self.feature_config.get('window_stride')
        self.number_of_bytes = self.feature_config.get('number_of_bytes')

        self.processed_path_suffix = (
            f"fg_{self.feature_generator}_"
            f"Wsize_{self.window_size}_"
            f"Wstride_{self.window_stride}_"
            f"nb_{self.number_of_bytes}"
        )

        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)
            self.logger.info("Output directory created successfully")

    def generate(self, values: np.ndarray, labels: pd.DataFrame, labeling_schema) -> pd.DataFrame:
        self.logger.info("Generating features using sliding window...")

        X, y = list(), list()

        # Loop of the entire data set
        for i in range(values.shape[0]):
            # Compute a new (sliding window) index
            start_ix = i*(self.window_stride)
            end_ix = start_ix + self.window_size - 1 + 1

            # If index is larger than the size of the dataset, we stop
            if end_ix >= values.shape[0]:
                break

            # Get a sequence of data for x
            seq_X = values[start_ix:end_ix]

            # Get a squence of data for y
            tmp_seq_y = labels[start_ix : end_ix]

            # Labeling schema
            seq_y = labeling_schema(tmp_seq_y)

            # Append the list with sequences
            X.append(seq_X)
            y.append(seq_y)
        
        x_array = np.array(X, dtype='float32')
        y_array = pd.DataFrame(y, columns=['label'])

        self.logger.info("[INFO] Feature generation complete.")
        self.logger.info(f"[INFO] Generated features shape: {x_array.shape}")
        self.logger.info(f"[INFO] Generated labels: {y_array['label'].value_counts()}")

        return x_array, y_array
    
    def get_output_path(self):
        """
        Get the output path for the processed data.
        """
        return f"{self.processed_path}/X_{self.phase}_{self.processed_path_suffix}.npz", \
               f"{self.processed_path}/y_{self.phase}_{self.processed_path_suffix}.csv"
