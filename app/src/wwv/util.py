import torch
import onnx
import onnxruntime
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
import logging 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())



class OnnxExporter:

    def __init__(self, model, cfg, output_dir):
        
        self.cfg = cfg
        self.model_name = cfg.model_name 
        self.model = model
        self.output_dir =output_dir
        self.model.eval()
        assert not self.model.training, "Model not in inference mode before exporting to onnx format"
        # Input to the model
        batch_size = 1
        # Get expected input dims from config cfg.processing_output_shape = (40, 241)
        self.x_in = torch.randn(batch_size, 1, 40, 241, device="cpu")

        logger.info(f"Input for model tracing: {self.x_in.shape}")
        self.x_out = self.model(self.x_in)
        logger.info(f"Output given input for model tracing: {self.x_out.shape}")
        self.onnx_model_path=None


    def verify(self):
        model = onnx.load(self.onnx_model_path)
        onnx.checker.check_model(model)


    def to_numpy(self,tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
            

    # Export the model
    def __call__(self):
        print("self.output_dir", self.output_dir)
        output_path = self.output_dir + "/model.onnx"
        print("self.output_path", output_path)

        logger.info(f"Onnx model output path: {output_path}")

        model = self.model
        x_dummy =  self.x_in 
        torch.onnx.export(model=model,                                       # model being run
                         args=x_dummy,                                       # model input (or a tuple for multiple inputs)
                         f=output_path,                                      # where to save the model (can be a file or file-like object)
                         export_params=True,                                 # store the trained parameter weights inside the model file
                         opset_version=15,                                   # Only certain operations are available, 17 includes FFTs and IFFTs
                         do_constant_folding=True,                           # whether to execute constant folding for optimization
                         input_names = ['input_mfcc', 'dummy_input'],        # the model's input names
                         output_names = ['output_wwp'],                      # the model's output names
                         dynamic_axes={'input_mfcc' : {0 : 'batch_size'},    # variable length axes
                                        'output_wwp' : {0 : 'batch_size'}}) 
        self.onnx_model_path = output_path
        self.verify()
        self.inference_session()
        return self 


    def inference_session(self):
        ort_session = onnxruntime.InferenceSession(str(self.onnx_model_path),
                                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # compute ONNX Runtime output prediction
        logger.info(f"ort_session: {ort_session.__dict__}")
        logger.info(f"ort_session.get_inputs(): {ort_session.get_inputs()}")
        ort_inputs = {ort_session.get_inputs()[0].name: self.to_numpy(self.x_in)}

        logger.info(f"ort_inputs {ort_inputs}")
        ort_outs = ort_session.run(None, ort_inputs)
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(self.to_numpy(self.x_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        logger.info("Exported model has been tested with ONNXRuntime, and the result looks good!")

        