import numpy as np
import torch
import random
from models.base_model import BaseModel

from torch import nn

class SeqWatch(BaseModel):
    def __init__(self, w_size, input_size=58, hidden_size=32, num_layers=2, p=0.5):
        super(SeqWatch, self).__init__()
        self.w_size = w_size

        # Encoder
        self.encoder_rnn_0 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=p)

        # Decoder
        self.decoder_rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=p, batch_first=True)

        # Out
        self.fc = nn.Linear(in_features=hidden_size, out_features=input_size)

    def encoder_forward(self, x):
        # shape of x: (N, window)

        outputs, (hidden, cell) = self.encoder_rnn_0(x)

        return hidden, cell

    def decoder_forward(self, x, hidden, cell):
        # shape of x: (N) but we want (1, N)
        x = x.unsqueeze(1)

        outputs, (hidden, cell) = self.decoder_rnn(x, (hidden, cell))
        # shape of outputs: (N, 1, hidden_size)

        predictions = self.fc(outputs.squeeze(1))
        # shape of outputs: (N, 1, output_size)

        return predictions, hidden, cell

    def forward(self, x, teacher_force_ratio=0.5):
        # (N, seq, n_bytes)
        x = x.transpose(1, 0)
        # (seq, N, n_bytes)

        target = x.clone()

        target_len, batch_size, target_size = x.shape
        outputs = torch.zeros(target_len, batch_size, target_size, device=x.device)

        # Encoder forward
        hidden, cell = self.encoder_forward(x)
        
        x = target[0]

        outputs[0] = x

        for t in range(1, target_len):
            output, hidden, cell = self.decoder_forward(x, hidden, cell)
            # output shape: (N, output_size)

            outputs[t] = output
            x = target[t] if random.random() < teacher_force_ratio else output

        x = outputs.transpose(0, 1)

        return x
    
    def featuring(self, values: np.ndarray, _):
        # y = values.clone()

        return values, values
