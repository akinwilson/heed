'''
Processing, training, data, model and augmentation configuration 
definition
'''
import numpy as np 
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Tuple,List
from pathlib import Path 


@dataclass 
class HTSwinCfg:
    '''for htSwin hyperparamater'''
    window_size: int = 8
    spec_size: int =  256
    patch_size: int = 4 
    stride: Tuple[int] = (2, 2)
    num_head: List = field(default_factory= lambda : [4,8,16,32])
    dim: int = 96
    num_classes: int = 2
    depth: List = field(default_factory= lambda : [2,6,6,4])


@dataclass
class ResNetCfg:
    '''Residual network architectural parameters'''
    num_blocks: List = field( default_factory = lambda : [8, 8, 36, 3] )
    

@dataclass
class SignalCfg:
    ''' Parameters for signal processing of HTSwin '''
    sample_rate : int = 16000
    audio_duration : int = 2
    clip_samples : int = int(sample_rate * audio_duration)
    window_size : int = 512
    hop_size  : int= 256
    mel_bins : int = 64
    fmin : int = 50
    fmax : int = 14000
    shift_max : int = int(clip_samples * 0.5)
    enable_tscam : bool = field(default=True, metadata= { "help":  "Enbale the token-semantic layer"})


@dataclass 
class FittingCfg:
    batch_size: int  = 16
    learning_rate: float = 1e-3 
    max_epoch : int = 500
    num_workers : int  = 18
    lr_scheduler_epoch : List  =  field(default_factory=  lambda : [10,20,30])
    lr_rate : List = field(default_factory = lambda :  [0.02, 0.05, 0.1])
    es_patience: int = 25
    train_bs : int = 16 
    val_bs : int  = 16
    test_bs : int = 16



@dataclass
class DataPathCfg:
    root_data_dir : str  = field(
        metadata={
            "help": "Root directory of training, validation and testing csv files"
        }
    )

    model_name : str = field(
        metadata={
            "help": "Name of model. Will be used to create model_dir"
        }
    )
    root_model_dir : str  = field(
        metadata={
            "help": "Root directory for outputs of training process"
        }
    )

    model_dir : str = field(
        init=False,
        metadata={
            "help": "Directory for outputs of training process. Will be created after init"
        }
    )

    
    def __post_init__(self):
        expected_files = ["train.csv", "test.csv", "val.csv"]
        if not all(x in os.listdir(self.root_data_dir) for x in expected_files):
            raise FileNotFoundError(f"Expected: {expected_files}\nTo exist in dir: {self.root_data_dir}.\nOnly found: {os.listdir(self.root_data_dir)}")

        model_dir = (Path(self.root_model_dir) / self.model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = str(model_dir)
        



    


class Config:
    def __init__(self, params):
        self.model_name = params['model_name']
        self.audio_feature = params['audio_feature']
        self.audio_duration = params['audio_duration']
        self.sr = params['sample_rate']
        self.audio_feature_param = params['audio_feature_param']
        self.augmentation = params['augmentation']
        self.augmentation_param =  params['augmentation_param']
        self.fit_param = params['fit_param']
        self.data_param = params['data_param']
        self.path = params['path']
        self.verbose = params['verbose']# True
        self.lr_scheduler_epoch = [100,200,300]
        self.lr_rates = [0.02, 0.05, 0.1] 

    @property
    def max_sample_len(self):
        return int(self.audio_duration * self.sr)

    @property 
    def processing_output_shape(self):
        attrname = 'audio_feature_param'
        if self.audio_feature == "mfcc":
            n_mfcc =  getattr(self,attrname)[self.audio_feature]['n_mfcc']
            hop_len = getattr(self,attrname)[self.audio_feature]['hop_length']

            time_step = int(np.around(self.max_sample_len / hop_len, 0))
            return  (n_mfcc, time_step)
        if self.audio_feature == "spectrogram":
            freq_bins =  getattr(self,attrname)[self.audio_feature]["n_mels"]
            hop_len =  getattr(self,attrname)[self.audio_feature]["hop_length"]
            time_steps = int(np.around(self.max_sample_len / hop_len, 0))
            return  (freq_bins,time_steps)
        if self.audio_feature == "pcm":
            return self.max_sample_len



# 75 *  0.02 / 16000 = a 