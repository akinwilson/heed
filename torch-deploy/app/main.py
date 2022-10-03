import base64
import os
import sys
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# import base64
import torch
import onnxruntime 
from config import CONFIG
import numpy as np 
import time 
import torchaudio 
from exception_handler import validation_exception_handler, python_exception_handler
from schema import *

# Initialize API Server
app = FastAPI(
    title="Wakeword Detection Verificator",
    description="Wakeword verification model for Sky Glass",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """
    logger.info(f"Running envirnoment: {CONFIG['ENV']}")
    logger.info(f"PyTorch using device: {CONFIG['DEVICE']}")
    # use  GPU
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
                                }
        )
    ]
    # use cpu
    providers = ['CPUExecutionProvider',]
    # use TensorRT
    providers = ['TensorrtExecutionProvider',]
    # get executioner from envirnoment
    providers = [os.environ.get("EXECUTION_PROVIDER")]
    

    inference_session = onnxruntime.InferenceSession(CONFIG['ONNX_MODEL_PATH'],
                                                     providers=providers)

    # add model and other preprocess tools too app state
    app.package = {
        "torch_model": None,
        "onnx_session": inference_session
    }


@app.post('/api/v1/predict',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
def do_predict(request: Request, body: InferenceInput):
    """
    Perform prediction on input data
    """

    logger.info('API predict called')
    logger.info(f'input: {body}')
    ################################################################
    #                   preprocess before onnx
    ################################################################
    # from base64str to array
    s = time.time()
    decoded_wav = base64.decodebytes(body.base64str)
    x = np.frombuffer(decoded_wav, dtype=np.int16)
    # (sample_len,)
    x = x[np.newaxis, ...]    
    # (channel_dim, sample_len)
    pad_up = 48000 - x.shape[-1] 
    X = np.pad(x, ((0,0), (0, pad_up)), 'constant')
    X = X[np.newaxis, ...]
    # (batch_dim, channel_dim, sample_len)    
    X  = X / np.iinfo(np.int16).max # normalise to range -1 to 1
    
    kwargs = {
        "window_fn": torch.hann_window,
        "wkwargs":
        { 
            "device": "cpu"
            }   
        }
    x_mfcc_trans = torchaudio.transforms.MFCC(melkwargs=kwargs)
    x_mfcc = x_mfcc_trans(torch.tensor(X).float()).numpy()
    ################################################################
    ################################################################
    # Inference with Onnx runtime
    result = app.package["onnx_session"].run(None, {"input_mfcc":x_mfcc.astype(np.float32)})
    f = time.time()
    # print("result", result)
    logger.info(f"Results from onnx: {result}")
    # Need to implement predictor method. logits currently being returned.
    ww_prob = result[0][0][0]
    # prepare json for returning
    return {
        "error": False,
        "result": {
            "wake_word_probability": ww_prob , 
            "prediction": 1.0 if ww_prob > CONFIG['DECISION_THRESHOLD'] else 0.0, 
            "false_alarm_probability": 1-ww_prob,
            "decision_threshold": CONFIG['DECISION_THRESHOLD'],
            "wwvm_version" : CONFIG['MODEL_VERSION'],
            "inference_time" : f-s 
            }
    }


@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }


if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8080,
                reload=True, debug=True)
                # , log_config="log.ini"
                # )
