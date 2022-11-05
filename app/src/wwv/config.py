'''
Processing, training, data, model and augmentation configuration 
definition
'''
import numpy as np 
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Tuple,List, Dict, DefaultDict
from collections import defaultdict
from pathlib import Path 



@dataclass 
class HTSwin:
    '''for htSwin hyperparamater'''
    window_size: int = 8
    spec_size: int =  256
    patch_size: int = 4 
    stride: Tuple[int] = (2, 2)
    num_head: List = field(default_factory= lambda : [4,8,16,8])
    dim: int = 48
    num_classes: int = 1
    depth: List = field(default_factory= lambda : [2,6,4,2])
    audio_feature: str = "pcm" # required for dataloading pipeline to extract correct features 
    model_name: str = "HSTAT" # 
    max_sample_len : int = 32000 
     
    model_dir: str =  "/home/akinwilson/Code/pytorch/output/model"

    # @dataclass
    # class Signal:
    #     ''' Parameters for signal processing of HTSwin '''
    #     sample_rate : int = 16000
    #     audio_duration : int = 2
    #     clip_samples : int = int(sample_rate * audio_duration)
    #     window_size : int = 512
    #     hop_size  : int= 256
    #     mel_bins : int = 64
    #     fmin : int = 50
    #     fmax : int = 14000
    #     shift_max : int = int(clip_samples * 0.5)
    #     enable_tscam : bool = field(default=True, metadata= { "help":  "Enbale the token-semantic layer"})



@dataclass
class Signal:
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
class ResNet:
    '''Residual network architectural parameters'''
    num_blocks: List = field( default_factory = lambda : [4, 6, 18, 3] )
    audio_feature: str = "mfcc" 
    model_name: str = "ResNet"

    
@dataclass
class Feature:

    audio_duration: float = 1.5
    sample_rate:int = 16000
    window_len:int = int( 0.040 * sample_rate )
    window_step:int = int( 0.020 *sample_rate ) 
    mel_coefficients:int = 16
    mel_filters:int = 40
    num_fft :int = 1024
    low_freq: float = 20.
    high_freq: float = 8000.
    melspec_kwargs: dict = field ( init=False,metadata={"help": "Parameters for mel spectrograme extraction of MFCC generation"}) 
                
    def __post_init__(self):
        self.melspec_kwargs =  {"n_fft": self.num_fft,
                                 "win_length": self.window_len,
                                 "hop_length": self.window_step,
                                 "n_mels": self.mel_filters,
                                 "onesided":True,
                                 "center": True,
                                 "pad_mode":"reflect",
                                 "f_min" : self.low_freq,
                                 "f_max" : self.high_freq} 


@dataclass 
class Fitting:
    batch_size: int  = 16
    learning_rate: float = 1e-3 
    max_epoch : int = 500
    num_workers : int  = 18
    lr_scheduler_epoch : List  =  field(default_factory=  lambda : [10,20,30])
    lr_rate : List = field(default_factory = lambda :  [0.02, 0.05, 0.1])
    es_patience: int = 25

    train_bs : int = 32
    val_bs : int  = 32
    test_bs : int = 32



@dataclass
class DataPath:
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
        



# @dataclass
# class Config:
#     model_name = params['model_name']
#     audio_feature = params['audio_feature']
#     audio_duration:int = 3
#     sr = 16000
#     audio_feature_param = params['audio_feature_param']
#     augmentation = params['augmentation']
#     augmentation_param =  params['augmentation_param']
#     fit_param = params['fit_param']
#     data_param = params['data_param']
#     path = params['path']
#     verbose = params['verbose']# True
#     lr_scheduler_epoch = [100,200,300]
#     lr_rates = [0.02, 0.05, 0.1] 

    
#     max_sample_len =  int(audio_duration * sr)

    # def processing_output_shape(self):
    #     attrname = 'audio_feature_param'
    #     if self.audio_feature == "mfcc":
    #         n_mfcc =  getattr(self,attrname)[self.audio_feature]['n_mfcc']
    #         hop_len = getattr(self,attrname)[self.audio_feature]['hop_length']

    #         time_step = int(np.around(self.max_sample_len / hop_len, 0))
    #         return  (n_mfcc, time_step)
    #     if self.audio_feature == "spectrogram":
    #         freq_bins =  getattr(self,attrname)[self.audio_feature]["n_mels"]
    #         hop_len =  getattr(self,attrname)[self.audio_feature]["hop_length"]
    #         time_steps = int(np.around(self.max_sample_len / hop_len, 0))
    #         return  (freq_bins,time_steps)
    #     if self.audio_feature == "pcm":
    #         return self.max_sample_len



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



# MODEL_DIR = "/home/akinwilson/Code/pytorch/output/model"  # location of root outputs
# DATA_DIR = "/home/akinwilson/Code/pytorch/dataset/keywords" # root location of meta training data csv files 
# LR_RANGE = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5][1] 
# BATCH_SIZE_RANGE = [1,2,16, 32, 64, 128, 256][2]
# EPOCH_RANGE = [1, 10, 30, 50, 100, 1000][1]
# ES_PATIENCE_RANGE = [1, 10, 20, 100, 200][2]
# MODELS = ["VecM5", "Resnet2vec1D","SpecResnet2D", "HSTAT", "DeepSpeech", "ResNet"][-1]
# AUDIO_FEATURE_OPT = ["spectrogram", "mfcc", "pcm"][1]
# PRETRAINED_MODEL_NAME_OR_PATH = "facebook/wav2vec2-base-960h"
# AUGS = False

# params = {
#     "audio_duration":3,
#     "sample_rate":16000,
#     "model_name": MODELS,
#     "verbose": False,
#     "path": {
#         "model_dir": MODEL_DIR,
#         "data_dir": DATA_DIR,
#         "pretrained_name_or_path": PRETRAINED_MODEL_NAME_OR_PATH
#         },
#     "fit_param": {"init_lr":LR_RANGE, "weight_decay":0.0001, "max_epochs":EPOCH_RANGE, "gamma": 0.1,"es_patience":ES_PATIENCE_RANGE}, 
#     "data_param":{"train_batch_size": BATCH_SIZE_RANGE, "val_batch_size": BATCH_SIZE_RANGE,"test_batch_size": BATCH_SIZE_RANGE}, 
#     "audio_feature": AUDIO_FEATURE_OPT,
#     "audio_feature_param": { "mfcc":{"sr":16000,"n_mfcc":20,"norm": 'ortho',"verbose":True,"ref":1.0,"amin":1e-10,"top_db":80.0,"hop_length":512,},
#                             "spectrogram":{"sr":16000, "n_fft":2048, "win_length":None,"n_mels":128,"hop_length":512,"window":'hann',"center":True,"pad_mode":'reflect',"power":2.0,"htk":False,"fmin":0.0,"fmax":None,"norm":1,"trainable_mel":False,"trainable_STFT":False,"verbose": True },
#                             "pcm": {}},
#     "augmentation":{'Gain': AUGS, 'PitchShift': AUGS, 'Shift': AUGS},
#     "augmentation_param":{"Gain": {  "min_gain_in_db":-18.0,"max_gain_in_db":  6.0,"mode":'per_example',"p":1,"p_mode":'per_example'},
#                         "PitchShift": {"min_transpose_semitones": -4.0, "max_transpose_semitones": 4.0,"mode":'per_example',"p":1,"p_mode":'per_example',"sample_rate":16000,"target_rate": None,"output_type": None,},
#                         "Shift":{ "min_shift":-0.5,"max_shift": 0.5,"shift_unit":'fraction',"rollover": True,"mode":'per_example',"p":1,"p_mode": 'per_example',"sample_rate": 16000,"target_rate":None,"output_type":None}},
#     }