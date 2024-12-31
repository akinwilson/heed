import matplotlib.pyplot as plt
import numpy as np 
import json 


import logging 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class Formatter:
    def __init__(self):
        '''
        Dont want to loose the values that generated the plots below
        this is a helper class to gather these values during plotting
        and saves them to a json file inside of the plots directory.  
        '''
        self.records = {}

    def formatter(self, metric, xt, yt, xv, yv):
        return {
            metric: {
                "train": {
                        "x":xt,
                        "y":yt
                        },
                "val": {
                        "x":xv,
                        "y":yv
                        }
                    }
        }
    def add(self, metric, xt, yt, xv, yv):
        record = self.formatter( metric, xt, yt, xv, yv)
        self.records = {**self.records, **record}

    def save_json(self, output_path):
        with open(output_path / "vals.json", "w", encoding='utf-8') as file:
            json.dump(self.records, file, ensure_ascii=False, indent=4)

        

class Plotter:
    def __init__(self,output_dir):
        self.store = Formatter()

        self.output_dir = output_dir
        self.metric_title = {
            "acc":"Accuracy", 
                "ttr":"TTR", 
                "ftr": "FTR"}

        self.metric_name = ["acc", "ttr", "ftr"]
    def get_loss(self,losses):
        x = [ e for (e,_) in losses]
        y = [ l for (_,l) in losses] 
        return x, y 

    def get_metrics(self, metrics_list,name:str):
        yt =    [ x[0][f"train_{name}"] for x in metrics_list]
        yv =  [ x[1][f"val_{name}"] for x in metrics_list]
        xt = np.arange(1, len(yt)+ 1,1).tolist()
        xv = np.arange(1, len(yv)+ 1,1).tolist()

        return xt, yt, xv, yv 

    def plot_learning_curves(self, train_losses, val_losses):
        # logger.info("Plotting learning curves")
        fig = plt.figure(figsize=(10,8))
        plt.title("Learning Curves")
        xt,yt = self.get_loss(train_losses)
        plt.plot(xt, yt, label = "train")
        xv,yv = self.get_loss(val_losses)
        plt.plot(xv, yv, label = "val")

        self.store.add("loss",  xt, yt, xv, yv)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        plt.savefig(self.output_dir / "lc.png")

    def plot_metric_curves(self, train_metrics, val_metrics):

        train_val = list(zip(train_metrics, val_metrics))

        fig, axs = plt.subplots(len(self.metric_name), 1,  figsize=(10, 18))
        fig.suptitle('Metric Curves', fontsize=12, y=0.94)
        for (ax, name) in zip(axs, self.metric_name):
            xt, yt, xv, yv  = self.get_metrics(train_val, name)
            ax.plot(xt,yt, label=f"train {name}")
            ax.plot(xv,yv, label=f"val {name}")
            self.store.add(name,  xt, yt, xv, yv)
            ax.set_title(self.metric_title[name], size=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(f"{self.metric_title[name]}")
            ax.legend()
            ax.grid()
        plt.savefig(self.output_dir / "metric.png")
    

    def save(self):
        output_path = self.output_dir
        self.store.save_json(output_path)

