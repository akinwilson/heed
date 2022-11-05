import torch.nn as nn 
import torch.nn.functional as F 

from .layers import * 

import torch 
import torch.nn  as nn 
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation



class LeeNetConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=kernel_size // 2, bias=False)
                              
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=1):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_size != 1:
            x = F.max_pool1d(x, kernel_size=pool_size, padding=pool_size // 2)
        return x



def do_mixup(x, mixup_lambda):
    """
    Args:
      x: (batch_size , ...)
      mixup_lambda: (batch_size,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x.transpose(0,-1) * mixup_lambda + torch.flip(x, dims = [0]).transpose(0,-1) * (1 - mixup_lambda)).transpose(0,-1)
    return out

def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled



class LeeNet(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.conv_block1 = LeeNetConvBlock2(1, 64, 3, 3)
        self.conv_block2 = LeeNetConvBlock2(64, 96, 3, 1)
        self.conv_block3 = LeeNetConvBlock2(96, 128, 3, 1)
        self.conv_block4 = LeeNetConvBlock2(128, 128, 3, 1)
        self.conv_block5 = LeeNetConvBlock2(128, 256, 3, 1)
        self.conv_block6 = LeeNetConvBlock2(256, 256, 3, 1)
        self.conv_block7 = LeeNetConvBlock2(256, 512, 3, 1)
        self.conv_block8 = LeeNetConvBlock2(512, 512, 3, 1)
        self.conv_block9 = LeeNetConvBlock2(512, 1024, 3, 1)

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = input[:, None, :]

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block2(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block3(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block4(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block5(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block6(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block7(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block8(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block9(x, pool_size=1)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict
