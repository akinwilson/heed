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

## Further improvements 
Currently, the [request-response](https://en.wikipedia.org/wiki/Request%E2%80%93response) cycle is being used to transfer the audio data from the frontend to the backend. Using the [websockets](https://en.wikipedia.org/wiki/WebSocket) protocol would allow for real-time communications between the server and client, this would more closely align with how such a model would be deployed in the real world. Django has a extension library, [channels](https://channels.readthedocs.io/en/latest/), which allows one to make use of this real-time communication protocol which websockets described.
    


