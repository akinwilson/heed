import torch
import torchaudio 
from torchvision import models
import torch.nn as nn

import torch.nn.functional as F
from transformers import AutoFeatureExtractor
from wwv.layer import AugmentationManager,Scaler, Standardisation, ProcessingLayer


import logging 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 






class Wav2vec(nn.Module):
    def __init__(self, cfg):
        '''
        Non-trainable feature extractor, based on pretrained model selected, not always Wav2Vec
        '''
        super().__init__()
        
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.path['pretrained_name_or_path'])
        self.cfg = cfg 
    def forward(self, x):
        feats = self.feature_extractor(x, sampling_rate=self.cfg.sr, return_tensors="pt").input_values.to(device)
        return torch.squeeze(feats, dim=0)




class VecM5(nn.Module):
    '''
    Model origins: 
        https://arxiv.org/pdf/1610.00087.pdf
    Meets 
    Model origin:
        https://arxiv.org/abs/2006.11477

    VecM5: wav2vec feature embedder with temporal convolutions for key-word spotting
    '''

    def __init__(self,cfg, n_input=1, n_output=1, stride=16, n_channel=32):
        super().__init__()
        self.cfg =cfg 
        self.extractor = Wav2vec(cfg)

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)


        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)

        #         self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        # self.bn4 = nn.BatchNorm1d(2 * n_channel)
        # self.pool4 = nn.MaxPool1d(4)

        self.conv5 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(2 * n_channel)
        self.pool5 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, 1)

    def forward(self, x):
        if self.cfg.verbose:
            logger.info(f"VecM5().foward() [in]: {x.shape}")

        x = self.extractor(x)
        if self.cfg.verbose:
            logger.info(f"VecM5().extractor() [out]: {x.shape}")

        x = self.conv1(x)
        if self.cfg.verbose:
            logger.info(f"VecM5().conv1() [out]: {x.shape}")
            
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.pool5(x)
        x = F.avg_pool1d(x, 2)
        x = x.permute(0,2,1)
        x = self.fc1(x)
        x = torch.squeeze(x)

        return x




class FullyConnected(torch.nn.Module):
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
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.hardtanh(x, 0, self.relu_max_clip)
        if self.dropout:
            x = torch.nn.functional.dropout(x, self.dropout, self.training)
        return x




class DeepSpeech(torch.nn.Module):
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
        self.bi_rnn = torch.nn.RNN(n_hidden, n_hidden, num_layers=1, nonlinearity="relu",batch_first=True, bidirectional=True)
        self.fc4 = FullyConnected(n_hidden, n_hidden, dropout)
        self.out = torch.nn.Linear(n_hidden, n_class)
        self.out = torch.nn.Linear(n_hidden, n_class)
        self.out = torch.nn.Linear(n_hidden, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            x, _ = self.bi_rnn(x)
            x = x[:, :self.n_hidden] + x[:, self.n_hidden :]
            x = self.fc4(x)
            x = self.out(x)
            x = torch.squeeze(x, dim=-1)
            # x = F.logsigmoid(x)
            return x










class Resnet2vec1D(nn.Module):
    pass 


# class SpecResnet2D(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         assert cfg.audio_feature in ["mfcc", "spectrogram"], f"{cfg.model_name} requires 2D inputs. Current input is 1D since cfg.audio_feature = {cfg.audio_feature}.\n\nChoose from audio_feature = ['spectrogram', 'mfcc'] to produce 2D inputs"
#         self.model = ResNet2D()

#     def forward(self, x):
#         return self.model(x)

class HSTAT(nn.Module):
    # https://github.com/RetroCirce/HTS-Audio-Transformer
    pass



MODELS = {
    "VecM5":VecM5, 
    "Resnet2vec1D": Resnet2vec1D,
    "HSTAT":HSTAT,
    "DeepSpeech": DeepSpeech
}



class Architecture(nn.Module):
    '''
    Architecture wrapper. 
    This class orchestrates all the processing and augmentation layers, followed by the architecture and returns a model 
    that can be trained and saved. 
    '''

    def __init__(self, cfg, training):
        super().__init__()
        self.cfg = cfg 
        # preprocessing layers 
        processing_layer_list =  [Scaler(cfg), Standardisation(cfg), AugmentationManager(cfg, training), ProcessingLayer(cfg)]
        self.processing_layer = torch.nn.Sequential(*processing_layer_list).to(device)
        #layers of the models
        ################################################################
        # Resnet2vec 
        # vecM5 
        #  Need to sort the architecture selections
        self.model = MODELS[cfg.model_name](cfg)
        # self.model # .to(device)
        ################################################################
        logger.info(f"{'-'*20}> Model name: {self.model.__class__.__name__}")
        logger.info(f"{'-'*20}> Model layers: {self.model}")
        logger.info(f"{'-'*20}> Trainingable params: {self.count_trainable_params()}")

    def count_trainable_params(self):
        traininable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        processing_trainable_params =  sum(p.numel() for p in self.processing_layer.parameters() if p.requires_grad)
        return  traininable_params + processing_trainable_params

        
    def forward(self, x):
        
        x_proccessed = self.processing_layer(x)
        logits = self.model(x_proccessed)
        
        if self.cfg.verbose:
            logger.info(f"Architecture().forward(x) [in]: {x.shape}")
            logger.info(f"Architecture().forward(x) processing_layer() [out]: {x_proccessed.shape}")
            logger.info(f"Architecture().forward(x)  model() [out]: {logits.shape}")

        return logits  
