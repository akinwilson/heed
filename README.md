# Notes for devs 

### Getting dataset 
Using DVC to store data remotely, such that you can fetch it when you try to run this repo. To get the dataset please make sure you have DVC installed. You can install it via pip or your system package manager. 

If you dont have relevant .csv files pulled via DVC, please just run the `/notebooks/get-data.ipynb` notebook. It will look for those files pull by DVC and create the train, val and test split, along with the csv files. 

### Getting a model 

At the moment, the easiest way for you to get a model.onnx file to test the endpoint is via running the first couple of cells of `/notebooks/arch-testing.ipynb`. You will have a directory produces called `/output` in the root directory of the repository. In there you will find the `model.onnx` that is needed to be placed into the `/torch-deploy/model` directory, for the deployment to work. 


### Building deployment container
Run: 
`docker build . -f torch-deploy/Dockerfile -t serve:latest`
From the directory level of this README.md file. The build context needs access to the model definition at the moment and install the python package locally. 
This will build the docker images

Run:
`docker build . -f torch-deploy/Dockerfile.tensorrt -t serve:latest`
To Build the image locally.
**Note** You will need to move a `model.onnx` into the `/torch-deploy/model` directory if you wish to deploy a model. 

Running the image iteractively 
`docker run --gpus all -p 8080:8080 -it serve:latest /bin/bash`
<br>
**Note** This will require Nvidia docker. You will enter the container at `/workspace`. 

Enter the command: `../app && python main.py` 
To start the server. Obviously you can start the server with single command  
Go to http://0.0.0.0:8080/docs to test out the enpoint via the swagger UI. 

To do:
3rd October 2022
- [x] Get one architecture working, as in fitting, and saving 
- [ ] parameterised the feature input dimensions, and pass this to the onnx
  exporter class during training
- [ ] Review https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/model.html#Wav2Vec2Model, test exporting the feature extractor of huggingface as part of the model architecture
- [ ] Normalisation approach: apply PCEN https://github.com/daemon/pytorch-pcen
- [x] change the output of networks to be logits let Routine function handle 
- [x] use nn.crossEntropy with raw logits -> not needed 
- [x] Make sure to adjust the Exporting of the model to onnx to include the correct input signature 
- [ ] Bench mark and stress test the TensorRT backend GPU (cuda) and CPU (MLAS) 
- [ ] Check back for when OpSet 17 is in stable torch, want to try to export
  feature extraction as part of model architecture. torch.ftt ops should be
supported in opset 17. 
- [ ] Create a presentation as to why 
- [ ] From onnx to TFlite -> implement basic converted and test against chris
  butcher
- [ ] factor out environment variables from docker container, place in file and
  load in at runtime docker run --env-args ... 
- [ ] Add more models to model collection, try finding span prediction dataset.
