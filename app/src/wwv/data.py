import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio as ta 
import torch
import torchaudio 
import logging

from wwv.layer import Scaler
import copy 


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu") 


class DataCollator:

    def __call__(self, batch):
        x = [ x for (x,_) in batch ]
        y = [ y for (_,y) in batch ]

        x_batched = torch.stack(x).float()
        y_batched = torch.stack(y).float()
        return {
        "x": x_batched,
        "y": y_batched
        }


class Padder:
    def __init__(self, cfg_model):
        self.cfg = cfg_model

    def __call__(self, x:torch.tensor) -> torch.tensor:
        padding = torch.tensor([0.0]).repeat([1,self.cfg.max_sample_len - x.size()[-1]])
        x_new = torch.hstack([x, padding])
        x_new = x_new.to(device) 
        assert x_new.size()[-1] == self.cfg.max_sample_len, f"Incorrect padded length, Should be {self.cfg.max_sample_len}, got {x_new.size()[-1]}"            
        return x_new # (1 ,1 , pad_to_len)



class AudioDataset(Dataset):
    def __init__(self,
                df_path,
                cfg_model,
                cfg_feature):
        self.df = pd.read_csv(df_path)

        self.x_pad = Padder(cfg_model)
        self.x_scale = Scaler()
        kwargs = {"window_fn": torch.hann_window,"wkwargs":{"device": device}}
        melkwargs = {**kwargs, **cfg_feature.melspec_kwargs}

        self.x_mfcc = torchaudio.transforms.MFCC(melkwargs=melkwargs)
        self.x_melspec = torchaudio.transforms.MelSpectrogram(**melkwargs)

        self.cfg_model = cfg_model
        self.cfg_feature = cfg_feature


    def __len__(self):

        return len(self.df)
    

    def __getitem__(self, idx):
        y = self.df.loc[idx]['label']
        x_path = self.df.loc[idx]['wav_path']
        y =  torch.tensor(int(y), device=device)
        x,_ = ta.load(x_path)
        ##########################################################
        #   Need to mimic this inside the serving container
        ##########################################################
        x = self.x_scale(x)
        x = self.x_pad(x)
        if self.cfg_model.audio_feature == "mfcc":
            x = self.x_mfcc(x)
            n_mfcc = int( (self.cfg_feature.sample_rate * self.cfg_feature.audio_duration) / self.cfg_feature.window_step )
            x = x[:,:,:n_mfcc]
            
        elif self.cfg_model.audio_feature == "spectrogram":
             x = self.x_melspec(x).transpose(1,2)
        else:
            x = x
        ##########################################################
        return x,y


class AudioDataModule():  # pl.LightningDataModule):
    def __init__(self,df_path, cfg_model, cfg_fitting, cfg_feature):
        super().__init__()

        # the DataPath data class makes sure the files below are present on init in the root directory. 
        self.train_df_path = df_path  + "/train.csv"
        self.val_df_path =  df_path  + "/val.csv"
        self.test_df_path =  df_path  + "/test.csv"

        self.cfg_model = cfg_model

        self.cfg_fitting = cfg_fitting
        self.cfg_feature = cfg_feature
        self.pin_memory =  False # True if torch.cuda.is_available() else False 
        

        # get input shape to network 
        dummpy_ds = self.test_dataloader()
        
        x = next(iter(dummpy_ds))
        input_shape = tuple(x['x'].shape[1:])
        input_shape = copy.deepcopy(input_shape)
        self.input_shape = input_shape
        del dummpy_ds 


    def train_dataloader(self):
        ds_train = AudioDataset(df_path=self.train_df_path,cfg_model= self.cfg_model,  cfg_feature=self.cfg_feature) # apply_augmentation)
        return DataLoader(ds_train,
                          batch_size=self.cfg_fitting.train_bs,
                          shuffle=True,
                          drop_last=True,
                          pin_memory= self.pin_memory,
                          collate_fn= DataCollator())

    
    
    def val_dataloader(self):
        ds_val = AudioDataset(df_path=self.val_df_path,  cfg_model= self.cfg_model, cfg_feature=self.cfg_feature)
        return  DataLoader(ds_val,
                          batch_size=self.cfg_fitting.val_bs,
                          shuffle=True,
                          drop_last=True,
                          pin_memory= self.pin_memory,
                          collate_fn= DataCollator())
    
    
    def test_dataloader(self):
        ds_test = AudioDataset(df_path=self.test_df_path,cfg_model= self.cfg_model, cfg_feature=self.cfg_feature)
        return  DataLoader(ds_test,
                          batch_size=self.cfg_fitting.test_bs,
                          shuffle=True,
                          drop_last=True,
                          pin_memory= self.pin_memory,
                          collate_fn= DataCollator())






