import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,3"
from wwv.Architecture.ResNet.model import ResNet
from wwv.Architecture.HTSwin.model import HTSwinTransformer
from wwv.Architecture.DeepSpeech.model import DeepSpeech

import torch.nn.functional as F 
from pytorch_lightning import Trainer
import torch.nn.functional as F 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor

from wwv.data import AudioDataModule
from wwv.util import OnnxExporter
from wwv.routine import Routine

class Predictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits =self.model(x)
        pred = F.sigmoid(logits)
        return pred 


class Fitter:

    def __init__(self, model, cfg_model, cfg, data_path="/home/akinwilson/Code/HTS-Audio-Transformer") -> None:
        self.model = model
        self.cfg_model = cfg_model
        self.cfg_fitting = cfg.Fitting()
        self.cfg_signal = cfg.Signal()
        self.cfg_feature = cfg.Feature()
        self.data_path = cfg.DataPath(data_path, self.cfg_model.model_name, self.cfg_model.model_dir)
        self.data_module = None
    def setup(self):
        data_module = AudioDataModule(self.data_path.root_data_dir,
                                    cfg_model=self.cfg_model,
                                    cfg_feature=self.cfg_feature,
                                    cfg_fitting=self.cfg_fitting)

        train_loader =  data_module.train_dataloader()
        val_loader =  data_module.val_dataloader()
        test_loader =  data_module.test_dataloader()
    
        return data_module, train_loader, val_loader, test_loader


    def get_callbacks(self):
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        early_stopping = EarlyStopping(mode="min", monitor='val_loss', patience=self.cfg_fitting.es_patience)
        checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                                dirpath=self.data_path.model_dir,
                                                save_top_k=1,
                                                mode="min",
                                                filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}-{val_ttr:.2f}-{val_ftr:.2f}')
        callbacks = [checkpoint_callback, lr_monitor, early_stopping]
        return callbacks 


    def __call__(self):
        logger = TensorBoardLogger(save_dir=self.data_path.model_dir, version=1, name="lightning_logs")
        Model = self.model

        if self.cfg_model.model_name == "HSTAT":
            kwargs = { "spec_size":self.cfg_model.spec_size,
                "patch_size":self.cfg_model.patch_size,
                "in_chans":1,
                "num_classes":self.cfg_model.num_classes,
                "window_size":self.cfg_model.window_size,
                "cfg_signal":self.cfg_signal, 
                "depths":self.cfg_model.depth,
                "embed_dim":self.cfg_model.dim,
                "patch_stride":self.cfg_model.stride,
                "num_heads": self.cfg_model.num_head}
        else:
            kwargs = {"num_blocks":self.cfg_model.num_blocks,"dropout":0.2}
        

        data_module, train_loader, val_loader, test_loader = self.setup()

        input_shape = data_module.input_shape

        model = Model(**kwargs)
        routine = Routine(model, self.cfg_fitting, self.cfg_model)
        trainer = Trainer(accelerator="gpu",
                        devices=3,
                        strategy='dp',
                        sync_batchnorm = True,
                        logger = logger, 
                        default_root_dir=self.data_path.model_dir,
                        callbacks=self.get_callbacks(),
                        num_sanity_val_steps = 2,
                        resume_from_checkpoint = None, 
                        gradient_clip_val=1.0,
                        fast_dev_run=False)

        # PATH  = "/home/akinwilson/Code/pytorch/output/model/ResNet/epoch=18-val_loss=0.15-val_acc=0.95-val_ttr=0.92-val_ftr=0.03.ckpt"                  
        trainer.fit(routine, train_dataloaders=train_loader, val_dataloaders=val_loader) # ,ckpt_path=PATH)
        trainer.test(dataloaders=test_loader)
        return input_shape, trainer 




if __name__ == "__main__":
    import torch 
    import torch.nn as nn 
    import wwv.config as cfg  

    torch.cuda.is_available()
    cfg_fitting = cfg.Fitting()
    cfg_signal = cfg.Signal()
    cfg_feature = cfg.Feature()
    cfg_model = cfg.HTSwin() # cfg.ResNet(), cfg.DeepSpeech()
    
    model = {"HSTAT": HTSwinTransformer,"ResNet":ResNet,"DeepSpeech": DeepSpeech}[cfg_model.model_name]

    fitter =Fitter(model, cfg_model, cfg)
    input_shape, fitted = fitter()
    model = fitted.model.module.module.model

    predictor = Predictor(model)
    OnnxExporter(model=predictor,
                model_name=fitter.cfg_model.model_name, 
                input_shape=input_shape,
                output_dir=fitter.data_path.model_dir, op_set=12)()
