from channels.generic.websocket import AsyncWebsocketConsumer
import requests
import matplotlib.pyplot as plt
import numpy as np

# import opuslib

import random

# import json
import base64
import io
import urllib

# import time
# import asyncio

# can reference container via service name, notice: it is the port internally to the serving container
# which we are sending requests to. Shouldnt really be a post request, since we dont change the state of the application at all.
# but current application server is configured to be a post request
URL_KWS_SERIVCE = "http://serving:80/api/v1/predict"


plt_colours = ["b", "g", "r", "c", "m", "y"]


def plot(audio_bytes=None, c=random.sample(plt_colours, k=1)[0]):
    fig, ax = plt.subplots()
    x = np.linspace(0, 2 * np.pi, 1000)
    y = [np.sin(_x) for _x in x]
    ax.plot(x, y, "*", color=c)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    # print("plot() CONSUMER: ", uri)
    return uri


# def iterate_plot():
#     for _ in range(10):
#         time.sleep(1)
#         t = loader.get_template("heed/index.html")
#         clr = random.sample(plt_colours, k=1)[0]
#         c = {"image_uri": plot(c=clr)}
#         yield t.render(c)


# import pyaudio as pa

# RATE = 16_000
# CHANNELS = 1
# CHUNK = 16_000  # determined by the javascript code on front-end
# FORMAT = pa.paInt16
# p = pa.PyAudio()

# stream_parser = p.open(
#     rate=RATE, channels=CHANNELS, format=FORMAT, frames_per_buffer=CHUNK, output=True
# )
from pyogg import OpusDecoder
import pyogg


class KWSConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        await self.accept()

    async def receive(self, bytes_data):
        # https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Audio_codecs#opus
        # print("raw_btyes: ", bytes_data)
        odecoder = OpusDecoder()
        odecoder.set_channels(1)
        odecoder.set_sampling_frequency(16_000)
        # print(dir(odecoder))
        try:
            print("Decoded bytes: ", odecoder.decode(bytes_data))
        except pyogg.pyogg_error.PyOggError:
            print("Corrupted stream apparently")
            pass
        # data = stream_parser.read(bytes_data)
        # print("Array: ", array)
        # print(f"pyaudio data: {data}")
        # sample = np.frombuffer(bytes_data, dtype=np.float32)  # np.int16)
        print(f"len of bytes: {len(bytes_data)}")
        # print("np array: ", sample.shape)

        # self.send(bytes_data)

        # self.send(text_data=BLUE_PLOT_B64)

    async def get_spot_result(self, data):
        # data better be base64 encoded
        print(f"get_spot_result(data) ->  data: {data}")
        result = requests.post(URL_KWS_SERIVCE, data=data)
        print("result.status_code: ", result.status_code)
        print("result: ", result)
        return result

    # {
    #         "keyword_probability": ww_prob,
    #         "prediction": 1.0 if ww_prob > CONFIG["DECISION_THRESHOLD"] else 0.0,
    #         "false_alarm_probability": 1 - ww_prob,
    #         "decision_threshold": CONFIG["DECISION_THRESHOLD"],
    #         "wwvm_version": CONFIG["MODEL_VERSION"],
    #         "inference_time": f - s,
    #     }

    async def spot(self, data):
        print(f"spot data: {data}")
        result = await self.get_spot_result(data)
        print("results from spot(): ", result)
        if "channel" in data:
            spot_result = data["channel"]["alternatives"][0]["spot"]

            if spot_result:
                await self.send(spot_result)

    async def response(self, data=None):
        """
        function responds to received audio stream via producing a plot of
        of the predicted values
        """
        pass
