import numpy as np
import torch.nn as nn
from torch.nn.utils import weight_norm

from models.base_model import BaseModel


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, T, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.T = T

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        if T == 2:
            self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation))
            self.chomp2 = Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)

            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                    self.conv2, self.chomp2, self.relu2, self.dropout2)

        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        if self.T == 2:
            nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, T, dropout=0.25):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, T=T, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TNCAnomalyDetector(BaseModel):
    def __init__(self, input_size=None, hidden_size=1, levels=1, kernel_size=3, T=1, dropout=0.3):
        super(TNCAnomalyDetector, self).__init__()

        self.input_size = input_size
        num_channels = [hidden_size] * (levels - 1) + [input_size]

        self.tcn = TemporalConvNet(num_inputs=input_size, num_channels=num_channels, kernel_size=kernel_size, T=T, dropout=dropout)
        self.linear = nn.Linear(input_size, input_size)

    def forward(self, x):
        # (N, seq, n_bytes)
    
        x_output = self.tcn(x.transpose(1, 2))
        x_output = self.linear(x_output.transpose(1, 2))
        
        return x_output

    def featuring(self, values: np.ndarray, _):
        # Apply TCN to predict the next packet
        
        X = values[:, :-1, :]
        y = values[:, 1:, :]

        return X, y