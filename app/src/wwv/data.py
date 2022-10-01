import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio as ta 
import torch
import torchaudio 
import logging

from wwv.layer import Scaler

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu") 


class DataCollator:
    def __init__(self, cfg):
        self.cfg = cfg 

    def __call__(self, batch):
        x = [ x for (x,_) in batch ]
        y = [ y for (_,y) in batch ]

        x_batched = torch.stack(x).float() 
        # x_batched.squeeze_(dim=1) # removing audio channel dim
        y_batched = torch.stack(y).float()
        if self.cfg.verbose:
            logger.info(f"DataCollator().__call__ x_batched [out]: {x_batched.shape}")
            logger.info(f"DataCollator().__call__ y_batched [out]: {y_batched.shape}")
        # return dictionary for unpacking easily as args 

        return {
            "x": x_batched,
            "y": y_batched
            }


class Padder:
    def __init__(self, cfg):
        self.cfg = cfg 

    def __call__(self, x:torch.tensor) -> torch.tensor:
        padding = torch.tensor([0.0]).repeat([1,self.cfg.max_sample_len - x.size()[-1]])
        x_new = torch.hstack([x, padding])
        x_new = x_new.to(device) 
        assert x_new.size()[-1] == self.cfg.max_sample_len, f"Incorrect padded length, Should be {self.cfg.max_sample_len}, got {x_new.size()[-1]}"
        if self.cfg.verbose:
            logger.info(f"Padder().__call__ x [out]: {x_new.shape}")
            # x_new = torch.unsqueeze(x_new, dim=1)
            # logger.info(f"Padder() unsqueeze [out]: {x_new.shape}")
            
        return x_new # (1 ,1 , pad_to_len)




class AudioDataset(Dataset):
    def __init__(self,
                df_path,
                cfg):
        self.df = pd.read_csv(df_path)

        self.x_pad = Padder(cfg)
        self.x_scale = Scaler(cfg)
        kwargs = {
            "window_fn": torch.hann_window,
            "wkwargs":
            { 
                "device": device
                }
                
            }
        self.x_mfcc = torchaudio.transforms.MFCC(melkwargs=kwargs)
        self.cfg = cfg


    def __len__(self):

        return len(self.df)
    

    def __getitem__(self, idx):
        y = self.df.loc[idx]['label']
        x_path = self.df.loc[idx]['wav_path']
        y =  torch.tensor(int(y), device=device)
        x,_ = ta.load(x_path)
        x = self.x_scale(x)
        x = self.x_pad(x)
        x = self.x_mfcc(x)
        return x,y


class AudioDataModule():  # pl.LightningDataModule):
    def __init__(self,train_df_path, val_df_path, test_df_path,cfg):
        super().__init__()

        self.train_df_path = train_df_path
        self.val_df_path =  val_df_path
        self.test_df_path =  test_df_path
        self.cfg = cfg 
        self.pin_memory =  False # True if torch.cuda.is_available() else False 
        


    def train_dataloader(self):
        ds_train = AudioDataset(df_path=self.train_df_path,cfg=self.cfg ) # apply_augmentation)
        return DataLoader(ds_train,
                          batch_size=self.cfg.data_param['train_batch_size'],
                          shuffle=True,
                          drop_last=True,
                          pin_memory= self.pin_memory,
                          collate_fn= DataCollator(self.cfg))

    
    
    def val_dataloader(self):
        ds_val = AudioDataset(df_path=self.val_df_path, cfg=self.cfg)
        return  DataLoader(ds_val,
                          batch_size=self.cfg.data_param['val_batch_size'],
                          shuffle=True,
                          drop_last=True,
                          pin_memory= self.pin_memory,
                          collate_fn= DataCollator(self.cfg))
    
    
    def test_dataloader(self):
        ds_test = AudioDataset(df_path=self.test_df_path,cfg=self.cfg)
        return  DataLoader(ds_test,
                          batch_size=self.cfg.data_param['test_batch_size'],
                          shuffle=True,
                          drop_last=True,
                          pin_memory= self.pin_memory,
                          collate_fn= DataCollator(self.cfg))




##############################################################################################################
# Way to provide metadata 
# for config file 
# unused 
##############################################################################################################

from pydantic import validate_arguments
from dataclasses import dataclass, field
# from typing import Dict, List, Optional


### Data class for hyperparameters. I use pydantic to make sure the typs are enforced and checked before being passed to the trainer. 
@validate_arguments
@dataclass
class Hyperparameters:
    beta_min: float = field(
        metadata={
            "help": "Adam Optimizer lower bound hyperparameter on coefficients used for computing running averages of gradient and its square"
        }
    )

    beta_max: float = field(
        metadata={
            "help": "Adam optimizer upper bound hyperparameter on coefficients used for computing running averages of gradient and its square"
        }
    )
    eps: float = field(
        metadata={
            "help": "Term added to the denominator to improve numerical stability"
        }
    )
    lr: float = field(
        metadata={
            "help": "Optmizer's learning rate (initial)"
        }
    ) 
    weight_decay: float = field(
        metadata={
            "help": "Optmizer's weight decay (L2 penalty)"
        }
    ) 
    
    hidden_dim: int = field(
        metadata={
            "help": "architectural parameter; hidden dimension width"
        }
    ) 
    batch_size: int = field(
        metadata={
            "help": "training routine parameter; batch size for parallel processing"
        }
    ) 
    max_epochs: int = field(
        metadata={
            "help": "training routine parameter; maximum epochs"
        }
    ) 
    num_classes: int = field(
        metadata={
            "help": "Training data parameter; number of target classes, needed by the F1 metric calculation"
        }
    ) 
        
        











