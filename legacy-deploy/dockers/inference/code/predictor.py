from __future__ import print_function
import base64
import flask
import json
import requests
import time
from os import environ

DECISION_THRESHOLD = 0.5
MODEL_VERSION = environ["MODEL_VERSION"]

TFS_URL = "http://localhost:8501/v1/models/model:predict"

app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    return flask.Response(
        response=json.dumps(""), status=200, mimetype="application/json"
    )


@app.route("/invocations", methods=["POST"])
def transformation():
    content_type = flask.request.content_type

    if content_type == "application/json":
        request_data = flask.request.json
    else:
        return flask.Response(
            response="This application only accepts content-type application/json",
            status=415,
            mimetype="text/plain",
        )

    start_time = time.time()
    processed_input = _input_handler(request_data)
    response = _get_prediction(processed_input)
    # print(f"@@@@@@@@@@@@ response {response}")
    output = _output_handler(response, start_time)

    return flask.Response(response=output, status=200, mimetype="application/json")


def _input_handler(request_data):
    bytes_str = base64.b64decode(request_data["content"])

    # Wouldn't work if someone tried to send a WAVE file with a 46 byte header, but that would be a fringe case
    if b"WAVE" in bytes_str[:44]:
        bytes_str = bytes_str[44:]

    model_input = (
        base64.b64encode(bytes_str).decode("utf-8").replace("+", "-").replace("/", "_")
    )
    return json.dumps({"signature_name": "serving_default", "inputs": [model_input]})


def _output_handler(response, start_time):
    ww_probability = response["outputs"][0][0]
    return json.dumps(
        {
            "prediction": 1 if ww_probability >= DECISION_THRESHOLD else 0,
            "false_alarm_probability": str(1 - ww_probability),
            "wake_word_probability": str(ww_probability),
            "decision_threshold": DECISION_THRESHOLD,
            "wwvm_version": MODEL_VERSION,
            "inference_time": time.time() - start_time,
        }
    )


def _get_prediction(processed_input):
    headers = {"content-type": "application/json"}
    response = requests.post(TFS_URL, data=processed_input, headers=headers)

    return response.json()
