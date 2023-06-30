import torch
import onnx
import onnxruntime
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
import torch.nn as nn
import torch.nn.functional as F

# from tensorflow import lite
# import tensorflow as tf
# import tensorflow_model_optimization as tfmot
# from onnx_tf.backend import prepare

from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)


class Predictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        pred = F.sigmoid(logits)
        return pred


class CallbackCollection:
    def __init__(self, cfg_fitting, data_path) -> None:
        self.cfg_fitting = cfg_fitting
        self.data_path = data_path

    def __call__(self):
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        early_stopping = EarlyStopping(
            mode="min", monitor="val_loss", patience=self.cfg_fitting.es_patience
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.data_path.model_dir,
            save_top_k=2,
            save_last=True,
            mode="min",
            filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}-{val_ttr:.2f}-{val_ftr:.2f}",
        )

        callbacks = {
            "checkpoint": checkpoint_callback,
            "lr": lr_monitor,
            "es": early_stopping,
        }
        # callbacks = [checkpoint_callback, lr_monitor, early_stopping]
        return callbacks


class OnnxExporter:
    def __init__(self, model, model_name, output_dir, input_shape, op_set=17):

        self.model_name = model_name
        self.model = model
        self.output_dir = output_dir
        self.op_set = op_set
        self.model.eval()
        assert (
            not self.model.training
        ), "Model not in inference mode before exporting to onnx format"
        # Input to the model
        batch_size = 1
        shape = (batch_size,) + input_shape
        # Get expected input dims from config cfg.processing_output_shape = (40, 241)
        self.x_in = torch.randn(shape, device="cpu")

        logger.info(f"Input for model tracing: {self.x_in.shape}")
        self.x_out = self.model(self.x_in)
        logger.info(f"Output given input for model tracing: {self.x_out.shape}")
        self.onnx_model_path = None

    def verify(self):
        model = onnx.load(self.onnx_model_path)
        onnx.checker.check_model(model)

    def to_numpy(self, tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # Export the model
    def __call__(self):
        print("self.output_dir", self.output_dir)
        output_path = self.output_dir + "/model.onnx"
        print("self.output_path", output_path)

        logger.info(f"Onnx model output path: {output_path}")

        model = self.model
        x_dummy = self.x_in
        torch.onnx.export(
            model=model,  # model being run
            args=x_dummy,  # model input (or a tuple for multiple inputs)
            f=output_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=self.op_set,  # Only certain operations are available, 17 includes FFTs and IFFTs
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input_mfcc", "dummy_input"],  # the model's input names
            output_names=["output_wwp"],  # the model's output names
            dynamic_axes={
                "input_mfcc": {0: "batch_size"},  # variable length axes
                "output_wwp": {0: "batch_size"},
            },
        )
        self.onnx_model_path = output_path
        self.verify()
        self.inference_session()
        return self

    def inference_session(self):
        ort_session = onnxruntime.InferenceSession(
            str(self.onnx_model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # compute ONNX Runtime output prediction
        # logger.info(f"ort_session: {ort_session.__dict__}")
        # logger.info(f"ort_session.get_inputs(): {ort_session.get_inputs()}")
        ort_inputs = {ort_session.get_inputs()[0].name: self.to_numpy(self.x_in)}

        # logger.info(f"ort_inputs {ort_inputs}")
        ort_outs = ort_session.run(None, ort_inputs)
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(
            self.to_numpy(self.x_out), ort_outs[0], rtol=1e-03, atol=1e-05
        )
        logger.info(
            "Exported model has been tested with ONNXRuntime, and the result looks good!"
        )


# class TfliteConverter:

#     def __init__(self, in_path, out_path, out_lite_path, out_lite_quant_path, test_loader, quantise=True):
#          self.in_path =  in_path
#          self.out_path = out_path
#          self.out_lite_path = out_lite_path
#          self.out_lite_quant_path = out_lite_quant_path
#          self.test_loader = test_loader
#          self.quantise=quantise


#     def get_torch_representative_dataset(self, test_loader):
#         representative_x = []
#         representative_y = []
#         for batch in test_loader:
#             x = batch['x']
#             y = batch['y']
#             representative_x.append(x)
#             representative_y.append(y)


#         x = torch.vstack(representative_x)
#         y = torch.stack(representative_y).view(-1,1)
#         return x,y


#     def torch_to_tf_dataset(self, x, y):
#         tf_feats = tf.convert_to_tensor(x.numpy())
#         tf_labels = tf.convert_to_tensor(y.numpy())

#         dataset = tf.data.Dataset.from_tensor_slices((tf_feats, tf_labels))
#         dataset = dataset.concatenate(dataset)
#         return dataset


#     def __call__(self):
#         # load from onnx and convert to tf
#         onnx_model = onnx.load(self.in_path)  # load onnx model
#         tf_rep = prepare(onnx_model)  # prepare tf representation
#         tf_rep.export_graph(self.out_path)  # export the model

#         # init convert
#         converter = lite.TFLiteConverter.from_saved_model(self.out_path)
#         converter.optimizations = [tf.lite.Optimize.DEFAULT]
#         # convert model in memory
#         tflite_model = converter.convert()
#         # save converted in-memory model
#         with open(self.out_lite_path, "wb") as file_handle:
#             file_handle.write(tflite_model)

#         if self.quantise:
#             # convert torch test set into tf dataset for quantisation purposes
#             test_loader = self.test_loader
#             x, y = self.get_torch_representative_dataset(test_loader)
#             representative_dataset = self.torch_to_tf_dataset(x,y)
#             # quantise the model
#             converter = lite.TFLiteConverter.from_saved_model(self.out_path)
#             converter.optimizations = [tf.lite.Optimize.DEFAULT]

#             converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#             converter.inference_input_type = tf.float32
#             converter.inference_output_type = tf.float32

#             converter.representative_dataset = representative_dataset

#             tflite_quant_model = converter.convert()
#             with open(self.out_quant_lite_path, "wb") as file_handle:

#                 file_handle.write(tflite_quant_model)
