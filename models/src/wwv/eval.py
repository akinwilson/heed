import torch


import logging 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class Metric:
    def __init__(self, y_hat, y):
        """
        y_hat and y are expected to be lists.

        m = Metric(y_hat, y)() 
            returns object with all outcomes and metrics as attrs
        e.g.
        m.__dict__ =
            {'y_hat': tensor([0., 0., 1., 1.]),
            'y': tensor([0., 1., 1., 0.]),
            'tp': 1,
            'tn': 1,
            'fn': 1,
            'fp': 1,
            'acc': 0.5,
            'ttr': 0.5,
            'ftr': 0.5} 
        """

        self.y_hat = y_hat #  torch.squeeze(torch.tensor(y_hat))
        self.y = y #  torch.squeeze(torch.tensor(y))

        # self.cfg = cfg 
        # if cfg.verbose:
        # logger.info(f"Metric().__init__  y_hat: {self.y_hat.shape}")
        # logger.info(f"Metric().__init__ y: {self.y.shape}")

        self.tp = 0
        self.tn = 0 
        self.fn = 0 
        self.fp = 0 
        self.acc = 0
        self.ttr = 0 
        self.ftr = 0

    def confusion(self):
        x = list(zip(self.y_hat, self.y))
        # logger.info(f"Confusion: {x}")
        
        fp = len([  (y_hat_val, y_val)  for (y_hat_val, y_val) in x if (y_hat_val, y_val) == (1.,0.)])
        tp = len([  (y_hat_val, y_val)  for (y_hat_val, y_val) in x if (y_hat_val, y_val) == (1.,1.)])
        fn = len([  (y_hat_val, y_val)  for (y_hat_val, y_val) in x if (y_hat_val, y_val)== (0.,1.)])
        tn = len([  (y_hat_val, y_val)  for (y_hat_val, y_val) in x if (y_hat_val, y_val) == (0.,0.)])
        self.tp = tp 
        self.tn = tn
        self.fp = fp
        self.fn = fn 
        # if self.cfg.verbose:
        try:
            self.acc = (tp + tn )/ (tn + tp +fp +fn )
        except ZeroDivisionError:
            self.acc = 0
        try:   
            self.ttr = tp / (tp+fn)
        except ZeroDivisionError:
            self.ttr = 0
        try:
            self.ftr = fp / (fp + tn)
        except ZeroDivisionError:
            self.ftr = 0
         
        return self 

    def __call__(self):
        return self.confusion()