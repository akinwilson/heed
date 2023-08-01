import os
from dotenv import load_dotenv

model_name = "ResNet"
env_filepath = os.getenv("ENV_FILE_PATH", f"../env_vars/{model_name.lower()}/.dev.env")
print(f"Loading env vars from file: {env_filepath}")
load_dotenv(env_filepath)


import optuna
from optuna.integration import PyTorchLightningPruningCallback
from wwv.Architecture.ResNet.model import ResNet
from wwv.routine import Routine

import torch.nn.functional as F
from wwv.data import AudioDataModule
import wwv.config as cfg
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    ModelPruning,
)

from wwv.util import get_username
from wwv.Architecture.ResNet.model import ResNet
from wwv.Architecture.HTSwin.model import HTSwinTransformer
from wwv.Architecture.DeepSpeech.model import DeepSpeech
from wwv.Architecture.LeeNet.model import LeeNet
from wwv.Architecture.MobileNet.model import MobileNet

from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)


from wwv.util import change_username_of_dataset_fileloactions, get_username


import torch.optim as optim
import optuna
from optuna import Trial, TrialPruned
from optuna.trial import TrialState


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


cfg_model = STR_TO_MODEL_CFGS[model_name]
# select comp graph/model arch
model = STR_TO_MODELS[model_name]


# init the fitter <---- associated  data loaders and fitting routine to model

model = model
cfg_model = cfg_model
cfg_fitting = cfg.Fitting()
cfg_signal = cfg.Signal()
cfg_feature = cfg.Feature()


data_path = cfg.DataPath(
    f"/media/{get_username()}/Samsung_T5/data/audio/keyword-spotting",
    cfg_model.model_name,
    cfg_model.model_dir,
)


def setup():
    """
    Set up data module and loaders
    """
    data_module = AudioDataModule(
        data_path.root_data_dir,
        cfg_model=cfg_model,
        cfg_feature=cfg_feature,
        cfg_fitting=cfg_fitting,
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    return data_module, train_loader, val_loader, test_loader


# get loaders and datamodule to access input shape
data_module, train_loader, val_loader, test_loader = setup()

# get input shape for onnx exporting
input_shape = data_module.input_shape
# init model


# callback_dict = callbacks()
# callback_list = [v for (_, v) in callback_dict.items()]
number_devices = os.getenv("CUDA_VISIBLE_DEVICES", "1,").split(",")
try:
    number_devices.remove("")
except ValueError:
    pass


def get_callbacks(trial: optuna.trial.Trial):
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stopping = EarlyStopping(
        mode="min", monitor="val_loss", patience=cfg_fitting.es_patience
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=data_path.model_dir,
        save_top_k=1,
        mode="min",
        filename=f"trial_{trial.number}_"
        + "{epoch}-{val_loss:.2f}-{val_acc:.2f}-{val_ttr:.2f}-{val_ftr:.2f}",
    )
    callbacks = [checkpoint_callback, lr_monitor, early_stopping]
    return callbacks


class Objective:
    def __init__(self, gpu_queue):
        # queue to manage GPU ids
        self.gpu_queue = gpu_queue

    def objective_value(
        self, trial: optuna.trial.Trial, monitored_metric="val_acc"
    ) -> float:
        # print(f"Called objective with trial: {trial.__dict__}")
        # We optimize the number of layers, hidden units in each layer and dropouts.
        dropout = trial.suggest_float("dropout", 0.2, 0.5)

        kwargs = {
            "num_blocks": cfg_model.num_blocks,
            "dropout": cfg_model.dropout,
        }

        Model = STR_TO_MODELS[model_name]
        kwargs["dropout"] = dropout

        model = Model(**kwargs)
        # setup training, validating and testing routines for the model
        routine = Routine(model, cfg_fitting, cfg_model)

        callbacks = get_callbacks(trial) + [
            PyTorchLightningPruningCallback(trial, monitor=monitored_metric)
        ]

        logger = TensorBoardLogger(
            save_dir=data_path.model_dir,
            name="lightning_logs",
        )

        trainer = Trainer(
            accelerator="gpu",
            devices=4,  # len(number_devices),
            strategy="ddp",  # os.getenv("STRATEGY", "ddp"),
            sync_batchnorm=True,
            max_epochs=cfg_fitting.max_epoch,
            callbacks=callbacks,
            num_sanity_val_steps=2,
            logger=logger,
            gradient_clip_val=1.0,
            fast_dev_run=cfg_fitting.fast_dev_run,
        )

        hyperparameters = dict([("dropout", dropout)])
        trainer.logger.log_hyperparams(hyperparameters)

        trainer.fit(
            routine, train_dataloaders=train_loader, val_dataloaders=val_loader
        )  # ,ckpt_path=PATH)
        print(f"Finished fitting for trial: {trial.number}")
        val_acc = trainer.callback_metrics["val_acc"].item()
        return val_acc

    def __call__(self, trial):
        gpu_id = self.gpu_queue.get()

        obj_val = self.objective_value(trial)

        self.gpu_queue.put(gpu_id)

        return obj_val


# def objective(trial: optuna.trial.Trial, monitored_metric="val_acc") -> float:
#     # print(f"Called objective with trial: {trial.__dict__}")
#     # We optimize the number of layers, hidden units in each layer and dropouts.
#     dropout = trial.suggest_float("dropout", 0.2, 0.5)

#     kwargs = {
#         "num_blocks": cfg_model.num_blocks,
#         "dropout": cfg_model.dropout,
#     }

#     Model = STR_TO_MODELS[model_name]
#     kwargs["dropout"] = dropout

#     model = Model(**kwargs)
#     # setup training, validating and testing routines for the model
#     routine = Routine(model, cfg_fitting, cfg_model)

#     callbacks = get_callbacks(trial) + [
#         PyTorchLightningPruningCallback(trial, monitor=monitored_metric)
#     ]

#     logger = TensorBoardLogger(
#         save_dir=data_path.model_dir,
#         name="lightning_logs",
#     )

#     trainer = Trainer(
#         accelerator="gpu",
#         devices=1,  # len(number_devices),
#         strategy="ddp",  # os.getenv("STRATEGY", "ddp"),
#         sync_batchnorm=True,
#         max_epochs=cfg_fitting.max_epoch,
#         callbacks=callbacks,
#         num_sanity_val_steps=2,
#         logger=logger,
#         gradient_clip_val=1.0,
#         fast_dev_run=cfg_fitting.fast_dev_run,
#     )

#     hyperparameters = dict([("dropout", dropout)])
#     trainer.logger.log_hyperparams(hyperparameters)

#     trainer.fit(
#         routine, train_dataloaders=train_loader, val_dataloaders=val_loader
#     )  # ,ckpt_path=PATH)
#     print(f"Finished fitting for trial: {trial.number}")
#     val_acc = trainer.callback_metrics["val_acc"].item()
#     return val_acc


# optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
# lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
# optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

if __name__ == "__main__":
    study_name = "example-study"  # Unique identifier of the study.

    url = f"sqlite:///{data_path.model_dir}/{study_name}.db"
    direction = "maximize"  # or minimize

    num_trials = 100  # number of trials PER WORKER

    print("Configuring storage backend ... ")
    print(f"Using storage location {url}")

    storage = optuna.storages.RDBStorage(
        url=url,
        engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
    )

    PRUNING = True
    pruner = optuna.pruners.MedianPruner() if PRUNING else optuna.pruners.NopPruner()

    study = optuna.create_study(direction=direction, pruner=pruner, storage=storage)
    print(f"Created study with direction: {direction}")

    from multiprocessing import Manager
    from joblib import parallel_backend

    with Manager() as manager:
        gpu_queue = manager.Queue()
        num_gpus = 4
        for i in range(num_gpus):
            gpu_queue.put(i)

        with parallel_backend("multiprocessing", n_jobs=num_gpus):
            study.optimize(
                Objective(gpu_queue), n_trials=num_trials, timeout=600, n_jobs=1
            )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
