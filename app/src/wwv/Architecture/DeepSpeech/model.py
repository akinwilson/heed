import torch.nn as nn 
import torch 
from .layers import FullyConnected


class DeepSpeech(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        n_feature = cfg.max_sample_len 
        n_hidden = 2048*2
        n_class = 1
        dropout = 0.2
        self.n_hidden = n_hidden
        self.fc1 = FullyConnected(n_feature, n_hidden, dropout)
        self.fc2 = FullyConnected(n_hidden, n_hidden, dropout)
        self.fc3 = FullyConnected(n_hidden, n_hidden, dropout)
        self.bi_rnn = nn.RNN(n_hidden, n_hidden, num_layers=1, nonlinearity="relu",batch_first=True, bidirectional=True)
        self.fc4 = FullyConnected(n_hidden, n_hidden, dropout)
        self.out = nn.Linear(n_hidden, n_class)
        self.out = nn.Linear(n_hidden, n_class)
        self.out = nn.Linear(n_hidden, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            x, _ = self.bi_rnn(x)
            x_forward = x[:, :self.n_hidden] 
            x_backward = x[:, self.n_hidden :]
            x = x_forward + x_backward
            x = self.fc4(x)
            x = self.out(x)
            x = torch.squeeze(x, dim=-1)
            return x

