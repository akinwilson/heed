# heed 
![]( img/heed.jpg )
## Overview 
heed is a library which fits a collection of deep learning models trained on the task of [key-word spotting](https://en.wikipedia.org/wiki/Keyword_spotting) and locally deploy one via a webserver which provides a user interface for these trained models to be queried through, using an audio input device of the host machine.

##  Usage

To begin, clone the repository and run a script to check that you have all the required CLI tools:
```
./check_cli_tools.sh
```
Unfortunately, the dataset used in this application is private. So running the entire pipeline end to end may not work with a single command for you. Nethertheless, you can used your own [binary classification](https://en.wikipedia.org/wiki/Binary_classification) dataset from the audio domain to substitute for the one used in this repository, or remove the `fitting` service from the `docker-compose.yaml` file and use a pre-trained model that accompanies this repository. 


To spin up the model fitting job, serving and web server containers, please run:
```
docker-compose -f docker-compose.yaml up
```

## Running tests


## Further improvements 

## Notes for devs 

### Create environment
To develop the zoo of models available, create a virtual environment and whilst in it, run the command: 
`pip install -r requirements.txt` 
and then 
`pip install -e ./models` 
This will install the dependencies and python package that contains the logic for all the models, allowing you to import it like 
```python 
import kws
...
```



### Building fitting container 

Fit a model via running the script:

`fit.py`


You will have a directory produces called `/output` in the root directory of the repository. In there you will find the `model.onnx` that is needed to be placed into the `/deploy/model` directory, for the deployment to work. 


### Building serving container

To Build the image locally run: THIS IS CURRENTLY FAILING: [SEE HERE]https://stackoverflow.com/questions/79325937/audio-stream-how-to-decode-opus-format-being-streamed-from-browser-to-server)
`docker build . -f deploy/Dockerfile.tensorrt -t serve:latest`

To deploy the image and test the endpoint, run:


 `docker run --gpus all -p 8080:80 -e "WORKERS=1" -e "EXECUTION_PROVIDER=TensorrtExecutionProvider"  --name="rt_test" -it serve:latest`
 


**Note** You will need to move a `model.onnx` into the `/deploy/model` directory if you wish to deploy a model. 

Running the image iteractively 

`docker run --gpus all -p 8080:8080 -it serve:latest /bin/bash`

**Note** This will require Nvidia docker. You will enter the container at `/workspace`. 

From within the container, enter the command: `../app && python main.py` 
To start the server. Obviously you can start the server with single command rather than entering into the container, but the `entrypoint` of the dockerfile needs to be configured
Go to 
```
http://0.0.0.0:8080/docs
```
to test out the enpoint via the swagger UI. 



## To do: 14 Jan 2025
- [] develop frontend javascript to use mediaRecorder in browser to recorder 1.5 seconds of audio to send to backend service serving kws model. 
- [] Containerise the fitting of the models and make sure access is provided. 
- [] You need to be able to run the entire pipeline with docker-compose. The issue that is currently persisting is the decoding of the [opus]() which you are currently tracking via [here](https://stackoverflow.com/questions/79325937/audio-stream-how-to-decode-opus-format-being-streamed-from-browser-to-server)

- [] Figure out issue with tensorrt: you are currently tracking this issue via [here](https://github.com/onnx/onnx-tensorrt/issues/1009). Cannot use as execution provider CUDA, but not tensorrt for the speed up of merging the weights and biases. 
- [] figure out how to include the `kms` library in the fitting container. Its part of private repository so running -e  

- [ ] Review https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/model.html#Wav2Vec2Model, test exporting the feature extractor of huggingface as part of the model architecture
- [ ] Normalisation approach: apply PCEN https://github.com/daemon/pytorch-pcen
- [ ] From onnx to TFlite -> implement basic converted and test against chris
  butcher
- [ ] Integrate feature extraction into model architecture using opset 17 and torch nightly via https://github.com/qiuqiangkong/torchlibrosa TRIED AND TESTED: WORKS FOR OPSET 17



Research
- [ ] Representational learning autoencoder visualising embeddings via tensorbaord - build downstrain classifier based auto-encoded representation
- [ ] Synthetic data generation using variational autoencoder - Train classifier on both mix, purely synthetic and purely the original dataset and compare the results 
- [ ] Is it possible to to generator a KWS alogrithm with synthetic data generation? As in, is it possibly to generate dataset for any given key-word to be stopped?   
