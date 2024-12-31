# heed 
![]( img/heed.jpg )
## Overview 
heed is a library which fits and locally deploys a collection of deep learning models trained on the task of [key-word spotting](https://en.wikipedia.org/wiki/Keyword_spotting), a server is  created whcih provides a user interface for these trained models to be queried through, using an audio input device of the host machine.

## Installation 


##  Usage

## Running tests

## Further improvements 

## Citation 


# Notes for devs 

### Create environment
Create a virtual environment and whilst in it, run the command: 
`pip install -r requirements.txt` 
and then 
`pip install -e ./app` 
This will install the dependencies and python package that contains the ML code. 



### Getting dataset 
Using DVC to store data remotely, such that you can fetch it when you try to run this repo. To get the dataset please make sure you have DVC installed. You can install it via pip or your system package manager. 

run the command: 
`dvc pull`
This will create a directory called `/dataset/keywords` with the required data to train the model

If you dont have relevant .csv files pulled via DVC, please just run the `/notebooks/get-data.ipynb` notebook. It will look for those files pull by DVC (the audio files) and create the train, val and test split, along with the csv files. 


#### Accessing data without gmail login
You can grab the dataset and unzip it yourself from the URL: 

`https://cdn.edgeimpulse.com/datasets/keywords2.zip`

Just make sure to place it in the appropriate dirtory: 

`/dataset/keywords`

The tree structure for the directory should be as follows: 

```
├── dataset
│   └── keywords
│       ├── no
│       ├── noise
│       ├── test.csv
│       ├── train.csv
│       ├── unknown
│       ├── val.csv
│       └── yes
```

### Getting a model 

Fit a model via running the script:

`fit.py`


You will have a directory produces called `/output` in the root directory of the repository. In there you will find the `model.onnx` that is needed to be placed into the `/deploy/model` directory, for the deployment to work. 


### Building deployment container

To Build the image locally run:
`docker build . -f deploy/Dockerfile.tensorrt -t serve:latest`

To deploy the image and test the endpoint, run:


 `docker run --gpus all -p 8080:80 -e "WORKERS=1" -e "EXECUTION_PROVIDER=TensorrtExecutionProvider"  --name="rt_test" -it serve:latest`
 


**Note** You will need to move a `model.onnx` into the `/deploy/model` directory if you wish to deploy a model. 

Running the image iteractively 

`docker run --gpus all -p 8080:8080 -it serve:latest /bin/bash`


**Note** This will require Nvidia docker. You will enter the container at `/workspace`. 


Enter the command: `../app && python main.py` 
To start the server. Obviously you can start the server with single command  
Go to http://0.0.0.0:8080/docs to test out the enpoint via the swagger UI. 

## To do: 29th November 2022

- [x] Get one architecture working, as in fitting, and saving 
- [x] Extend model zoo
- [x] parameterised the feature input dimensions, and pass this to the onnx
  exporter class during training
- [ ] Review https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/model.html#Wav2Vec2Model, test exporting the feature extractor of huggingface as part of the model architecture
- [ ] Normalisation approach: apply PCEN https://github.com/daemon/pytorch-pcen
- [x] change the output of networks to be logits let Routine function handle 
- [x] use nn.crossEntropy with raw logits -> not needed 
- [x] Make sure to adjust the Exporting of the model to onnx to include the correct input signature 
- [x] Bench mark and stress test the TensorRT backend GPU (cuda) and CPU (MLAS) 
- [x] Check back for when OpSet 17 is in stable torch, want to try to export
  feature extraction as part of model architecture. torch.ftt ops should be
supported in opset 17. 
- [x] Create a presentation as to why 
- [ ] From onnx to TFlite -> implement basic converted and test against chris
  butcher
- [x] factor out environment variables from docker container, place in file and
  load in at runtime docker run --env-args
- [x] Add more models to model collection, try finding span prediction dataset.
- [ ] Add model architecture for localisation task
- [ ] Integrate feature extraction into model architecture using opset 17 and torch nightly via https://github.com/qiuqiangkong/torchlibrosa TRIED AND TESTED: WORKS FOR OPSET 17
Research
- [ ] Implement high-informational content sampling strategy; compare to vanilla strategy the fitting time and achieved metrics 
- [ ] Representational learning autoencoder visualising embeddings via tensorbaord - build downstrain classifier based auto-encoded representation
- [ ] Synthetic data generation using variational autoencoder - Train classifier on both mix, purely synthetic and purely the original dataset and compare the results 
- [ ] Summarise the  
