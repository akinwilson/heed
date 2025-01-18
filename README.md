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
once the fitting job has been completed (which should be apparent from the logs), the serving container will have a [interactive API documentation page](https://fastapi.tiangolo.com/features/#based-on-open-standards) accessible for end users to test out the inference API directly. This will be accessible from:

```
http://localhost:6000/docs
```
there will also be a webserver serving an application allowing end users to test out the inference API via recording their own clips and posting them to the API through a simple user interface. This will be accessible from:
```
http://localhost:7000
```



## Running tests
heed has three components; the `fitting`, `serving` and `webserver` parts. Each of these has its own suite of tests which can be run via:
```

```

## Further improvements 
Expanding the model zoo. Websockets for real-time stream predictions. see `website/dev.README.md`

## Notes for devs 
See the `dev.README.md`s in each subfolder; `models/`, `website/` and `serve/`, for more information. 

### Create environment for model development. 
To develop the zoo of models available, create a virtual environment and whilst in it, run the command: 
`pip install -r requirements.txt` 
and then 
`pip install -e ./models` 
This will install the dependencies and python package that contains the logic for all the models, allowing you to import it like 
```python 
import kws
...
```

## To do: 14 Jan 2025
- [x] develop frontend javascript to use mediaRecorder in browser to recorder 1.5 seconds of audio to send to backend service serving kws model. 
- [x] Containerise the fitting of the models and make sure access is provided.
- [ ] Standardise docker `ENTRYPOINT`. Want to be able to supply `command` in `docker-compose.yaml` such ports and host ips can be configured from one file, the `docker-compose.yaml` file. 

- [ ] connect serving model to frontend once a trained model has been created. 
- [ ] finish writing tests. 
- [ ] using `npm` and `webpack` and bootstrap to provide styling to the frontend. Follow [this](https://getbootstrap.com/docs/5.3/getting-started/webpack/) tutorial. This will require refactoring the current frontend directoris into a `src/` and `dist/` directories. But it will allow you in the future to bundle js and css modules together for a production env and localise all our editing into one location; `src/`.
- [ ] Clean the serving application. Currently, there is preprocessing occuring inside of the application, this shouldnt be happening at the API level. Preferable onnx handles the transformations too. 

- [ ] You need to be able to run the entire pipeline with docker-compose. Close to having end to end state. need to figure out the data loading issue. 

- [ ] Add information to readme on dataset directory structure expectations so user can easily use their own datasets to train models but dropping them into the correct location.  

- [ ] Figure out issue with tensorrt: you are currently tracking this issue via [here](https://github.com/onnx/onnx-tensorrt/issues/1009). Cannot use as execution provider CUDA, but not tensorrt for the speed up of merging the weights and biases. 
- [ ] Review https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/model.html#Wav2Vec2Model, test exporting the feature extractor of huggingface as part of the model architecture
- [ ] Normalisation approach: apply PCEN https://github.com/daemon/pytorch-pcen
- [ ] Integrate feature extraction into model architecture using opset 17 and torch nightly via https://github.com/qiuqiangkong/torchlibrosa TRIED AND TESTED: WORKS FOR OPSET 17



Research
- [ ] Representational learning autoencoder visualising embeddings via tensorbaord - build downstrain classifier based auto-encoded representation
- [ ] Synthetic data generation using variational autoencoder - Train classifier on both mix, purely synthetic and purely the original dataset and compare the results 
- [ ] Is it possible to to generator a KWS alogrithm with synthetic data generation? As in, is it possibly to generate dataset for any given key-word to be stopped?   
