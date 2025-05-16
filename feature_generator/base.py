import os
import pandas as pd
import numpy as np
import typing

from abc import ABC, abstractmethod

class FeatureGenerator(ABC):
    def __init__(self, config: typing.Dict):
        self.config = config

        self.processed_path = config.get("processed_path")

    @abstractmethod
    def load_raw(self) -> tuple[np.ndarray, pd.DataFrame]:
        pass

    @abstractmethod
    def generate(self, values: np.ndarray, labels: pd.DataFrame, labeling_schema) -> tuple[pd.DataFrame, np.ndarray]:
        pass

    @abstractmethod
    def save(self, x: np.ndarray, y: pd.DataFrame):
        pass

    def run(self):
        df = self.load_raw()
        processed = self.generate(df)
        self.save(processed)

    def load_processed(self, subset: int = None) -> tuple[np.ndarray, pd.DataFrame]:
        if not os.path.exists(self.x_path):
            self.run()

        X = np.load(self.x_path)
        X = X.f.arr_0

        y = pd.read_csv(self.y_path, index_col=0)

        if subset is not None:
            indices = np.random.choice(len(X), size=int(subset*len(X)), replace=False)
            X = X[indices]
            y = y.iloc[indices].reset_index(drop=True)

        return X, y
