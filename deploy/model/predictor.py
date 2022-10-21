import numpy as np 
import torch
import time 
import logging 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())



'''
Note that this approach requires the model_path to have 
the ENTIRE model at the location and all of its dependenies 
it is by no means the most efficient way of serving. 

Prefereable, we want to use the Onnx format and OnnxRuntime to 
serve predictions.  
'''



device = "cuda" if torch.cuda.is_available() else "cpu"

class Predictor:
    def __init__(self, model_path="model.pt", device=device):
        self.device = device
        self.model = torch.load(model_path, map_location=device)
    
    def predict(self, x:torch.tensor)-> np.ndarray:
        """
        param:
            x: x.shape = (batch_dim, channel_dim, time_series)     
                    e.g. (1,         1,        max_sample_len)
        return:
            prob 
        """
        # make sure the tensor is on the same device as the model 
        x = x.to(self.device)
        with torch.no_grad(): # dont do gradient calcs
            logits = self.model(x)
            # convert to probabilities and squueze inplace
            prob = torch.distributions.Bernoulli(logits=logits).probs.squeeze_()
            # detach from graph
            raw_prob = prob.cpu().numpy()
            return raw_prob

    
    def __call__(self, x:torch.tensor)-> float:
        return self.predict(x)



if __name__ == "__main__":
    logger.info("Loading model...")
    s = time.time()
    predictor = Predictor()
    logger.info(f"Device: {predictor.device}")
    f = time.time()
    logger.info(f"Loading model: {f-s:.5f}s")
    x = torch.rand(1000,1,48000)
    logger.info("Serve...")
    s = time.time()
    probs = predictor(x)
    f = time.time()
    logger.info(f"Inference time for batched input: {f-s:.5f}s")
    logger.info(f"No. prob:{len(probs)}")


    timings = []

    for _ in range(1000):
        x = torch.rand(1,1,48000)
        s =time.time()
        _ = predictor(x)
        f = time.time()
        timings.append(f-s)
    
    
    logger.info(f"Average inference time for unbatched input: {sum(timings) / len(timings):.5f}s")
    
    