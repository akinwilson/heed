
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class FullyConnected(nn.Module):
    """
    Args:
        n_feature: Number of input features
        n_hidden: Internal hidden unit size.

        Fully connected layer with clipping
    """

    def __init__(self, n_feature: int, n_hidden: int, dropout: float, relu_max_clip: int = 20) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(n_feature, n_hidden, bias=True)
        self.relu_max_clip = relu_max_clip
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.relu(x)
        x = F.hardtanh(x, 0, self.relu_max_clip)
        if self.dropout:
            x = F.dropout(x, self.dropout, self.training)
        return x

