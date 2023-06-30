import os
import sys
import logging
from dotenv import load_dotenv
from argparse import ArgumentParser

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# only set ENV_FILE_PATH during testing
# how can I get args.model_name.lower()

parser = ArgumentParser()
MODEL_NAMES = ["HSTAT", "ResNet", "DeepSpeech", "LeeNet", "MobileNet"]
parser.add_argument(
    "-m", "--model_name", type=str, default="ResNet", choices=MODEL_NAMES
)
args, _ = parser.parse_known_args()
# model_name = "ResNet".lower()
# env_filepath = os.getenv("ENVFILE_PATH", f"./env_vars/{model_name}/.dev.env")

# I only know args at main function
env_filepath = os.getenv(
    "ENV_FILE_PATH", f"./env_vars/{args.model_name.lower()}/.dev.env"
)

logger.info(f"Loading env vars from file: {env_filepath}")
load_dotenv(env_filepath)

# import all available models
from wwv.Architecture.ResNet.model import ResNet
from wwv.Architecture.HTSwin.model import HTSwinTransformer
from wwv.Architecture.DeepSpeech.model import DeepSpeech
from wwv.Architecture.LeeNet.model import LeeNet
from wwv.Architecture.MobileNet.model import MobileNet

from wwv.data import AudioDataModule
from wwv.util import OnnxExporter, CallbackCollection
from wwv.routine import Routine
import wwv.config as cfg

import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn

STR_TO_MODEL_CFGS = {
    "HSTAT": cfg.HTSwin(),
    "ResNet": cfg.ResNet(),
    "DeepSpeech": cfg.DeepSpeech(),
    "LeeNet": cfg.LeeNet(),
    "MobileNet": cfg.MobileNet(),
}
STR_TO_MODELS = {
    "HSTAT": HTSwinTransformer,
    "ResNet": ResNet,
    "DeepSpeech": DeepSpeech,
    "LeeNet": LeeNet,
    "MobileNet": MobileNet,
}


class Predictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        pred = F.sigmoid(logits)
        return pred


class Fitter:
    def __init__(
        self,
        model,
        cfg_model,
        cfg,
        data_path="/media/useraye/Samsung_T5/data/audio/keyword-spotting",
    ) -> None:
        self.model = model
        self.cfg_model = cfg_model
        self.cfg_fitting = cfg.Fitting()
        self.cfg_signal = cfg.Signal()
        self.cfg_feature = cfg.Feature()
        self.data_path = cfg.DataPath(
            data_path, self.cfg_model.model_name, self.cfg_model.model_dir
        )

    def setup(self):
        data_module = AudioDataModule(
            self.data_path.root_data_dir,
            cfg_model=self.cfg_model,
            cfg_feature=self.cfg_feature,
            cfg_fitting=self.cfg_fitting,
        )

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()

        return data_module, train_loader, val_loader, test_loader

    def callbacks(self):
        cfg_fitting = self.cfg_fitting
        data_path = self.data_path
        callback_collection = CallbackCollection(cfg_fitting, data_path)
        return callback_collection()

    def __call__(self):
        logger = TensorBoardLogger(
            save_dir=self.data_path.model_dir,
            name="lightning_logs",
        )
        Model = self.model

        if self.cfg_model.model_name == "HSTAT" or self.cfg_model.model_name == "hstat":
            kwargs = {
                "spec_size": self.cfg_model.spec_size,
                "patch_size": self.cfg_model.patch_size,
                "in_chans": 1,
                "num_classes": self.cfg_model.num_classes,
                "window_size": self.cfg_model.window_size,
                "cfg_signal": self.cfg_signal,
                "depths": self.cfg_model.depth,
                "embed_dim": self.cfg_model.dim,
                "patch_stride": self.cfg_model.stride,
                "num_heads": self.cfg_model.num_head,
            }

        elif (
            self.cfg_model.model_name == "ResNet"
            or self.cfg_model.model_name == "resnet"
        ):
            kwargs = {
                "num_blocks": self.cfg_model.num_blocks,
                "dropout": self.cfg_model.dropout,
            }

        elif (
            self.cfg_model.model_name == "LeeNet"
            or self.cfg_model.model_name == "leenet"
        ):
            kwargs = {
                "dropout": self.cfg_model.dropout,
                "channel_order_dims": self.cfg_model.channel_order_dims,
            }

        elif (
            self.cfg_model.model_name == "DeepSpeed"
            or self.cfg_model.model_name == "deepspeed"
        ):
            kwargs = self.cfg_model.__dict__
            kwargs = {}

        elif (
            self.cfg_model.model_name == "MobileNet"
            or self.cfg_model.model_name == "mobilenet"
        ):
            kwargs = self.cfg_model.__dict__

        # get loaders and datamodule to access input shape
        data_module, train_loader, val_loader, test_loader = self.setup()

        # get input shape for onnx exporting
        input_shape = data_module.input_shape
        # init model
        model = Model(**kwargs)
        # setup training, validating and testing routines for the model
        routine = Routine(model, self.cfg_fitting, self.cfg_model)
        # Init a trainer to execute routine
        callback_dict = self.callbacks()
        callback_list = [v for (_, v) in callback_dict.items()]
        number_devices = os.getenv("CUDA_VISIBLE_DEVICES", "1,").split(",")
        try:
            number_devices.remove("")
        except ValueError:
            pass

        trainer = Trainer(
            accelerator="gpu",
            devices=len(number_devices),
            strategy=os.getenv("STRATEGY", "ddp"),
            sync_batchnorm=True,
            logger=logger,
            max_epochs=self.cfg_fitting.max_epoch,
            callbacks=callback_list,
            num_sanity_val_steps=2,
            # resume_from_checkpoint=self.cfg_fitting.resume_from_checkpoint,
            gradient_clip_val=1.0,
            fast_dev_run=self.cfg_fitting.fast_dev_run,
        )

        # PATH  = "/home/akinwilson/Code/pytorch/output/model/ResNet/epoch=18-val_loss=0.15-val_acc=0.95-val_ttr=0.92-val_ftr=0.03.ckpt"
        # Trainer executes fitting; training and validating proceducres
        trainer.fit(
            routine, train_dataloaders=train_loader, val_dataloaders=val_loader
        )  # ,ckpt_path=PATH)

        if self.cfg_fitting.fast_dev_run:
            # issue with finding best weights path for in fast dev run using last model weights
            model_ckpt_path = callback_dict["checkpoint"].__dict__["last_model_path"]
        else:
            model_ckpt_path = callback_dict["checkpoint"].__dict__["best_model_path"]

        trainer.test(
            dataloaders=test_loader,
            ckpt_path=model_ckpt_path,
        )
        # Return the input_shapes and trainer of the model for exporting
        return input_shape, trainer


def main():

    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"Training model: {args.model_name}")
    # configs for trainer, features and signal normnalisation  etc. Some overflap
    # cfg_fitting = cfg.Fitting()
    # cfg_signal = cfg.Signal()
    # cfg_feature = cfg.Feature()
    # look into the config.py file so see what available model classes there are
    cfg_model = STR_TO_MODEL_CFGS[args.model_name]
    # select comp graph/model arch
    model = STR_TO_MODELS[args.model_name]
    # init the fitter <---- associated  data loaders and fitting routine to model
    fitter = Fitter(model, cfg_model, cfg)
    # Get back input_shapes and fitted_trainer from fitter
    # ********hacky way to get input shape ***************
    input_shape, fitted = fitter()

    # # In data parallelism, fitted_trainer contains model in nested structure.
    # # ********* just condition here or always train in dist env **********
    if os.getenv("STRATEGY", "ddp") == "ddp":
        model_fitted = fitted.model._modules["model"]
    else:
        model_fitted = fitted.model._modules["model"]

    # appending probability normalisation layer to model to convert logits to probs.
    # ******** fitting: training and validating, is more stable with these as outputs.
    # ******** See loss function
    predictor = Predictor(model_fitted)
    # exporter classifier via onnx
    # **** model filename will just be model.onnx.
    # Note: using op_set 12 default at the moment but can back feature extraction into
    # model with  opset 17 and torchlibrosa. Tested for Swin.
    # ******************** NOT TEST PREDICTIONS YET THOUGH  **********************
    # init the exporter.
    onnx_exporter = OnnxExporter(
        model=predictor,
        model_name=fitter.cfg_model.model_name,
        input_shape=input_shape,
        audio_feature=cfg_model.audio_feature,
        output_dir=fitter.data_path.model_dir,
        op_set=cfg_model.onnx_op_set,
    )
    # exporting the model baby!
    onnx_exporter()


if __name__ == "__main__":
    main()
    sys.exit(0)
