import torch 
import torch.nn.functional as F
from torch import optim
from pathlib import Path 
from datetime import datetime 
import json 
from torch.utils.tensorboard import SummaryWriter


import logging 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


from wwv.data import AudioDataModule
from wwv.Architecture import Architecture
from wwv.layer import AugmentationManager
from wwv.eval import Metric
from wwv.plot import Plotter
from wwv.util import OnnxExporter
from wwv.config import DataPath
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 



def get_label(tensor):
    # find most likely label for each element in the batch
    # print(f"get_label() [in] {tensor}")
    x_norm = F.sigmoid(tensor)
    # print(f"get_label() [out_sigmoid] {x_norm}")
    # logger.info(f"get_label() following sigmoid: {x_norm.shape}")
    # print("get_likely_index inside x_norm after squueze", torch.squeeze(x_norm).shape)
    tensor = torch.squeeze(x_norm)
    # print("(tensor>0.5).float() ", (tensor>0.5).float() )
    return (tensor>0.5).float() 


def log_device_info(data):
    device_found = []
    for (_,tensor) in data.items():
        device_found.append(tensor.device)
    logger.info(f"Device info {device_found}")



def train_one_step(model, data, optimizer):
    optimizer.zero_grad()
    # data = {k:v.to(device) for (k,v) in data.items()}
    x_dict = {k:v for (k,v) in data.items() if k == 'x'}
    # logger.info(f"Input shape: {x_dict['x'].shape}")
    y_hat = model(**x_dict)
    y_hat = y_hat.squeeze().view(-1)
    y = data['y'].squeeze()
    loss = F.binary_cross_entropy_with_logits(y_hat , y, reduction='mean')
    loss.backward()
    optimizer.step()
    return model , loss 


def train_one_epoch(model, data_loader, optimizer, cfg):
    # model put into training mode 
    all_predictions, all_targets = [ ], [ ]
    flatten =  lambda l: sum(l , [])
    model.train()
    total_loss = 0 
    for batch_idx, data in enumerate(data_loader):
        data = {k: v.to(device) for (k,v) in data.items()}
        # log_device_info(data)
        model, loss = train_one_step(model, data, optimizer)
        total_loss += loss.item()

        x_dict = {k:v for (k,v) in data.items() if k == 'x'}
        output = model(**x_dict)
        pred = get_label(output)
        # logger.info(f"train_one_epoch() model outputs {output.shape}")
        # logger.info(f"train_one_epoch() pred outputs {pred.shape}")
        all_predictions.append(pred.cpu().numpy().tolist())
        all_targets.append(data['y'].cpu().numpy().tolist())

    
    tot_preds = flatten(all_predictions)
    tot_trgs = flatten(all_targets)
    metrics = Metric(y_hat=tot_preds, y=tot_trgs ,cfg=cfg)()

    return model, total_loss, metrics.acc, metrics.ttr, metrics.ftr 



def validate_one_step(model, data):
    # data = {k: v.to(device) for (k,v) in data.items()}
    x_dict = {k:v for (k,v) in data.items() if k == 'x'}
    y_hat = model(**x_dict)
    y_hat = y_hat.squeeze()
    y = data['y'].squeeze()
    loss = F.binary_cross_entropy_with_logits(y_hat, y , reduction='mean')
    return model, loss 



def validate_one_epoch(model,  data_loader, cfg):

    all_predictions, all_targets = [ ], [ ]
    flatten =  lambda l: sum(l , [])

    model.eval()
    total_loss = 0
    for batch_idx, data in enumerate(data_loader):
        data = {k: v.to(device) for (k,v) in data.items()}
        # log_device_info(data)
        model, loss = validate_one_step(model, data)
        total_loss += loss.item()
        # data = {k: v.to(device) for (k,v) in data.items()}
        x_dict = {k:v for (k,v) in data.items() if k == 'x'}
        output = model(**x_dict)
        pred = get_label(output)
        # logger.info(f"validate_one_epoch() model outputs {output.shape}")
        # logger.info(f"validate_one_epoch() pred outputs {pred.shape}")

        all_predictions.append(pred.cpu().numpy().tolist())
        all_targets.append(data['y'].cpu().numpy().tolist())

    
    tot_preds = flatten(all_predictions)
    tot_trgs = flatten(all_targets)

    metrics = Metric(y_hat=tot_preds, y=tot_trgs, cfg=cfg)()
    return model, total_loss, metrics.acc, metrics.ttr, metrics.ftr 





def test(model, epoch, data_module, cfg):
    model.eval()
    test_loader = data_module.test_dataloader()
    
    all_targets, all_predictions = [], []
    
    for data in test_loader:

        data = {k:v.to(device) for (k,v) in data.items()}
        x_dict = {k:v for (k,v) in data.items() if k == 'x'}
        output = model(**x_dict)
        pred = get_label(output)
        all_predictions.append(pred.cpu().numpy().tolist())
        all_targets.append(data['y'].cpu().numpy().tolist())


    flatten =  lambda l: sum(l , [])
    tot_preds = flatten(all_predictions)
    tot_trgs = flatten(all_targets) 
    metrics = Metric(y_hat=tot_preds, y=tot_trgs, cfg=cfg)()

    logger.info("*"*30)
    logger.info("*"*30)
    logger.info(f"Epoch: {epoch}")
    logger.info(f"{'-'*20}> test_acc: {metrics.acc:.2f}")
    logger.info(f"{'-'*20}> test_ttr: {metrics.ttr:.2f}")
    logger.info(f"{'-'*20}> test_ftr: {metrics.ftr:.2f}")    

    return model, metrics.acc, metrics.ttr, metrics.ftr 



def check_es(hist, min_or_max='max', es_patience=5):
    extrema = lambda x : max(x) if min_or_max=='max' else min(x)
    return len(hist) - hist.index(extrema(hist)) >= es_patience , extrema(hist)
    
    
    

def main(cfg):
    # global model
    
    model = Architecture(cfg, training=True)
    if torch.cuda.device_count() > 1: # resources for distributed data parallel training available
        num_gpus = torch.cuda.device_count()
        model = torch.nn.DataParallel(model)
        logger.info(f"{num_gpus} GPUs available for training")
        logger.info(f"Scaling up batch size accordingly ... ")
        cfg.data_param  = {k: int(bs*num_gpus) for (k,bs) in cfg.data_param.items()}

    model = model.to(device)    
    optimizer = optim.Adam(model.parameters(),
                            lr=cfg.fit_param['init_lr'],
                             weight_decay=cfg.fit_param['weight_decay'])
    opt_class_name = type(optimizer).__name__ 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=cfg.fit_param['gamma']) 


    data_path = DataPath(cfg.path['data_dir'])

    train_df_path = data_path.root_dir + "/train.csv"
    val_df_path =  data_path.root_dir +  "/val.csv"
    test_df_path =  data_path.root_dir +  "/test.csv"
    logger.info(f"Loading data from path: {train_df_path}")
    logger.info(f"Loading data from path: {val_df_path}")
    logger.info(f"Loading data from path: {test_df_path}")
    logger.info(f"Initializing data loader ... ")
    data_module = AudioDataModule(train_df_path, val_df_path, test_df_path, cfg)
    early_stopping_condition_met = False 
    es_metric_history = []
    epoch = 0 

    # Saving model artefacts, plots etc.
    date = datetime.now().strftime("%Y_%m_%d_%I.%M.%S_%p")
    model_dir = Path(cfg.path['model_dir']) /  cfg.model_name / date

    logger.info(f"Creating directories ...")
    weights_dir  = model_dir / "state"
    tf_board_dir = model_dir / "tf_board"
    inference_dir = model_dir / "inference"
    plots_dir = model_dir / "plot"
    export_dir = model_dir / "export"

    (weights_dir).mkdir(parents=True, exist_ok=True)
    (tf_board_dir).mkdir(parents=True, exist_ok=True)
    (inference_dir).mkdir(parents=True, exist_ok=True)
    (plots_dir).mkdir(parents=True, exist_ok=True)
    (export_dir).mkdir(parents=True, exist_ok=True)


    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []

    tf_board_writer = SummaryWriter(log_dir=tf_board_dir)
    logger.info("Entering fitting routine...")
    for epoch in range(1, cfg.fit_param['max_epochs'] + 1):
        ###############################################################################################################
        # Training
        ###############################################################################################################
        train_loader = data_module.train_dataloader()
        model, total_train_epoch_losss,  train_acc, train_ttr, train_ftr  = train_one_epoch(model, train_loader, optimizer, cfg)

        
        metrics = {
            "train_acc": train_acc,
            "train_ttr": train_ttr,
            "train_ftr": train_ftr
            }
        train_metrics.append(metrics)
        train_loss = total_train_epoch_losss/len(train_loader)

        train_losses.append((epoch, train_loss))
        logger.info(f"Epoch: {epoch}")
        logger.info(f"{'-'*20}> train_loss: {train_loss}")
        logger.info(f"{'-'*20}> train_metrics: {metrics}")
        
        tf_board_writer.add_scalar("loss/train", train_loss, epoch)
        tf_board_writer.add_scalar("acc/train", train_acc, epoch)
        tf_board_writer.add_scalar("ttr/train", train_ttr, epoch)
        tf_board_writer.add_scalar("ftr/train", train_ftr, epoch)


        scheduler.step()
        ###############################################################################################################
        # validatingg
        ###############################################################################################################
        val_loader = data_module.val_dataloader()
    
        model, total_val_epoch_losss, val_acc, val_ttr, val_ftr = validate_one_epoch(model, val_loader, cfg)
        val_loss = total_val_epoch_losss/len(val_loader)
        val_losses.append((epoch, val_loss))
        metrics = {
            "val_acc": val_acc,
            "val_ttr": val_ttr,
            "val_ftr": val_ftr
            }
        val_metrics.append(metrics)
        logger.info(f"{'-'*20}> val_loss: {val_loss}")
        logger.info(f"{'-'*20}> val_metrics: {metrics}")

        
        tf_board_writer.add_scalar("loss/val", val_loss, epoch)
        tf_board_writer.add_scalar("acc/val", val_acc, epoch)
        tf_board_writer.add_scalar("ttr/val", val_ttr, epoch)
        tf_board_writer.add_scalar("ftr/val", val_ftr, epoch)

        ###############################################################################################################
        # Callbacks
        ###############################################################################################################
        es_metric_history.append(val_acc)
        early_stopping_condition_met, extrema_val = check_es(es_metric_history, es_patience=cfg.fit_param['es_patience'])
        if early_stopping_condition_met:
            logger.info("Early stopping condition met")
            break

    logger.info(f"Final epoch: {epoch}")
    
    model, test_acc, test_ttr, test_ftr = test(model, epoch, data_module, cfg)  
    
    tf_board_writer.add_scalar("acc/test", test_acc, epoch)
    tf_board_writer.add_scalar("ttr/test", test_ttr, epoch)
    tf_board_writer.add_scalar("ftr/test", test_ftr, epoch)

    # Call flush() method to make sure that all pending events have been written to disk.
    tf_board_writer.flush()
    # If you do not need the summary writer anymore, call close() method.
    tf_board_writer.close()



    model_state_filename = f"val-acc-{extrema_val:.2f}.pt"
    optimizer_state_filename = f"opt-{opt_class_name}-epoch-{epoch}.pt"
    inference_filename =  f"{cfg.model_name}.pt"

    # Before saving the model, lets re-init the augmentation manager and set the 
    # regime of training to be false, this will return the identity transformation
    # by the augmentation manager
    # Processing layer contains the Standardisation [0], augmentation transforms [1] 
    # and feature extraction layer [2].
    if isinstance(model, torch.nn.DataParallel):
        # How to access module in distributed setting
        model.module.processing_layer[1] = AugmentationManager(cfg, False)
    else:
         model.processing_layer[1] = AugmentationManager(cfg, False)

    # saving the configuration as json file
    with open(model_dir / "cfg.json", "w", encoding='utf-8') as file:
                json.dump(cfg.__dict__, file, ensure_ascii=False, indent=4)



    # saving states 
    OnnxExporter(model, cfg, export_dir)()
    torch.save(model.state_dict() , weights_dir / model_state_filename)
    torch.save(optimizer.state_dict() , weights_dir / optimizer_state_filename)
    # saving inference model 
    torch.save(model, inference_dir /  inference_filename)

    plotter = Plotter(output_dir=plots_dir)
    plotter.plot_learning_curves(train_losses, val_losses)
    plotter.plot_metric_curves(train_metrics, val_metrics )
    plotter.save()
    # return train_losses, val_losses, train_metrics, val_metrics 




