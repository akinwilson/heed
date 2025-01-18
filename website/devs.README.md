# Keyword spotting webserver application 

## Overview 
This is a django webserver application designed to record a short audio clip - around 2 seconds - on the frontend  and send this recording to the backend where a deep learning model performs a prediction on said audio clip and returns to the frontend the probability of spotting the keyword along with a plot of the wavform of the recording. 

## Usage
You will need to create a virtual environment, and install the dependencies with 

```
pip install -r requirements.txt
```

Run the server in **development** mode with

```
python manage.py runserver 127.0.0.1:8000
```

**Note**: Inference on the audio clip requires a model to be propped up ready to receive requests. Currently, this inference service is deployed using `docker-compose` as seen in the `docker-compose.yaml` in the above directly; see `serving` service. Running the web server alone without a model being served for inference will cause the `ajax` post which posts the audio clip to the backend to receive a response with an error.


