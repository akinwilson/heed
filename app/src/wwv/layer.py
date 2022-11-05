

import logging 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

import json 
import torch
import torch.nn as nn 
from nnAudio import features
from pathlib import Path 
from torch_audiomentations import Compose, Gain, PitchShift, Shift, Identity
import torchvision.transforms as T 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 



class Scaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("int16_max", torch.tensor([32767]).float())
        # self.cfg = cfg 


    def forward(self, x:torch.tensor):

        x_scaled = x / self.int16_max
        return x_scaled 


class Standardisation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        '''
        Class standardises input using statistic gathering during preprocessing
        '''
        self.cfg = cfg
        # stats_path = Path(cfg.path['data_dir']) / "stats.json"
        # with open(stats_path, "r") as file:
        #     stats_dict = json.loads(file.read())

        # for k in  stats_dict:
        #     # setattr(self, k, torch.tensor([stats_dict[k]], device=device).float())
        #     self.register_buffer(k, torch.tensor([stats_dict[k]]).float())

        self.register_buffer("mean", torch.tensor([0.03]).float())
        self.register_buffer("var", torch.tensor([0.5]).float())

    def forward(self, x:torch.tensor) -> torch.tensor:
        x_norm = ( x - self.mean ) / self.var**(1/2)
        if self.cfg.verbose:
            logger.info(f"Standardisation().foward() [in]: {x.shape}")
            logger.info(f"Standardisation().foward() [out]: {x_norm.shape}")
        return x_norm 




class AugmentationManager(nn.Module):
    def __init__(self, cfg, training):
        super().__init__()
        '''
        Augmentation class prodvides a transformation pipeline to be 
        appplied to the data during loading. These transforms should not 
        be part of the architecture, but rather the transforms of the data 
        '''
        self.cfg =cfg 
        self.training = training
        self.transforms = self.compose_augmentations()

    def compose_augmentations(self):
        augs = [k for (k,v) in self.cfg.augmentation.items() if v]
        augs_params = [v for (k, v) in self.cfg.augmentation_param.items() if k in augs]
        func_dict = {
            "Gain":Gain,
            "PitchShift":PitchShift,
            "Shift":Shift
            }
        logger.info(f"{'-'*20}> Augmentations to be applied: {augs}")

        if len(augs) == 0:
            transforms_list = [Identity(mode='per_example', p=1)]
        else:    
            transforms_list = [func_dict[k](**v) for (k,v) in zip(augs,augs_params)]
        if self.training: 
            transforms = Compose( transforms_list)
        else:
            logger.info(f"self.training={self.training}: -> Identity layer returned for AugmentationManager")
            transforms = Compose([Identity(mode='per_example', p=1)] )
        return transforms


    def forward(self, x):
        x_out  = self.transforms(x)
        if self.cfg.verbose:
            logger.info(f"AugmentationManager().foward() [in]: {x.shape}")
            logger.info(f"AugmentationManager().foward() [out]: {x_out.shape}")
        return x_out 



class ProcessingLayer(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.cfg =cfg
        layers = []
        kwargs = cfg.audio_feature_param[cfg.audio_feature]
        if cfg.audio_feature == "spectrogram":
            layers.append(features.MelSpectrogram(**kwargs))
            # layers.append(T.Resize(224)) # size expected by 2D ResNet 
        elif cfg.audio_feature == "mfcc":
            layers.append(features.MFCC(**kwargs))
            # layers.append(T.Resize(224)) # size expected by 2D ResNet

        # resize inputs
        # layers.append(transforms.RandomResizedCrop(224))
        self.net = torch.nn.Sequential(*layers)
        logger.info(f"{'-'*20}> Features to be extracted: {cfg.audio_feature}")
        logger.info(f"{'-'*20}> Feature dimensions: {cfg.processing_output_shape}")

    def forward(self, x:torch.tensor) -> torch.tensor:
        x_out = self.net(x)
        if self.cfg.verbose:
            logger.info(f"ProcessingLayer().foward() [in]: {x.shape}")
            logger.info(f"ProcessingLayer().foward() [out]: {x_out.shape}")
        return x_out 