import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn.preprocessing import OneHotEncoder

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.le = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def forward(self, x):
        raise NotImplementedError("Each model must implement the forward pass.")
    
    def featuring(self, values: np.ndarray, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        y_encoded = self.le.fit_transform(labels)

        return values, y_encoded
