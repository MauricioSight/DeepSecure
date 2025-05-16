import pandas as pd
import numpy as np
from typing import Tuple
from scapy.all import rdpcap, raw

from feature_generator.base import FeatureGenerator

class TOWIDSFeatureLoader(FeatureGenerator):
    def __init__(self, config, logger, feature_generator):
        super().__init__(config)
        self.logger = logger

        self.feature_generator = feature_generator

        self.raw_y_path = config.get('raw_y_path')
        self.raw_x_path = config.get('raw_x_path')

        x_path, y_path = self.feature_generator.get_output_path()
        self.x_path = x_path
        self.y_path = y_path

        self.feature_config = config.get('feature_config')
        self.number_of_bytes = self.feature_config.get('number_of_bytes')

    def run(self):
        self.logger.info("Starting feature pipeline...")

        # Step 1: Generate features
        x, y = self.generate()

        # Step 2: Save output
        self.save(x, y)

        self.logger.info("Pipeline complete.")

    def load_raw(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load raw data from the TOW-IDS dataset.
        The dataset consists of two files: one for labels and one for values.
        The labels file contains the class and label information, while the values file
        contains the raw packet data.
        The function reads both files, processes the packet data, and returns a DataFrame
        containing the processed data along with the corresponding labels.
        """

        self.logger.info("Loading raw data...")

        # 1. Load labels
        labels = pd.read_csv(self.raw_y_path, header=None, names=["index", "class", "label"])
        labels = labels.drop(columns=["index"])
        labels['label'] = labels['label'].replace(self.get_label_mapping())

        # 2. Load values
        raw_packets = rdpcap(self.raw_x_path)
        converted_packets_list = []
        for raw_packet in raw_packets:
            converted_packet = np.frombuffer(raw(raw_packet), dtype='uint8')

            converted_packet_len = len(converted_packet)
            if converted_packet_len < self.number_of_bytes:
                bytes_to_pad = self.number_of_bytes - converted_packet_len
                converted_packet = np.pad(converted_packet, (0, bytes_to_pad), 'constant')
            else:
                converted_packet = converted_packet[0:self.number_of_bytes]

            # Normalize the packet data dividing by 255
            converted_packet = converted_packet / 255.0

            converted_packets_list.append(converted_packet)
        values = np.array(converted_packets_list, dtype='float32')

        return values, labels

    def get_label_mapping(self):
        return {
            'Normal': 'Normal',
            'C_D': 'CAN DoS',
            'P_I': 'PTP Sync',
            'M_F': 'Switch MAC Flooding',
            'F_I': 'Frame Injection',
            'C_R': 'CAN Replay',
        }
    
    def labeling_schema(self, sequence: pd.DataFrame) -> bool:    
        seq_y = 'Normal'
        labels = self.get_label_mapping().values()
        labels = [label for label in labels if label != 'Normal']

        indexes = sequence['label'].value_counts().sort_values(ascending=False).reset_index()
        indexes_list = list(indexes['label'].values)

        set_attacks = set(labels)
        set_sequence_indexes = set(indexes_list)

        intersect = any(set_atk in set_sequence_indexes for set_atk in set_attacks)

        if intersect is True:
            attacks_mask = indexes['label'].isin(labels)
            indexes_attacks = indexes[attacks_mask]
            seq_y = indexes_attacks['label'].values[0]

        return seq_y
    
    def generate(self):
        """
        Generate features and labels from the raw data.
        The function uses the `load_raw` method to load the data, and then applies
        the labeling schema to generate the labels for each sequence of data.
        It returns a tuple containing the processed data and the corresponding labels.
        """
        values, labels = self.load_raw()

        self.logger.info(f"Data loaded successfully.")
        self.logger.info(f"Data shape: {values.shape}")
        self.logger.info(f"Label shape: {labels.shape}")
        self.logger.info(f"Converted packets: {values[0][:2]}")
        self.logger.info(f"Labels: {labels['label'].value_counts()}")
        self.logger.info('')

        x, y = self.feature_generator.generate(values, labels, self.labeling_schema)
        
        return x, y
    
    def save(self, x: np.ndarray, y: pd.DataFrame):
        """
        Save the generated features and labels to CSV files.
        The function saves the features to a CSV file and the labels to a separate CSV file.
        """

        self.logger.info("Saving features and labels...")

        x_path, y_path = self.feature_generator.get_output_path()

        np.savez(x_path, x)
        y.to_csv(y_path)

        self.logger.info(f"Features saved to {x_path}.")

    def load_processed(self, subset = None) -> Tuple[np.ndarray, pd.DataFrame]:
        if subset is not None:
            self.logger.info(f"Loading processed data with subset of {subset}%")
        return super().load_processed(subset)
