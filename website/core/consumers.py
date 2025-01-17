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
import json
from channels.generic.websocket import AsyncWebsocketConsumer
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class AudioConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Accept the WebSocket connection
        # self.room_group_name = "audio_stream"
        # await self.channel_layer.group_add(
        #     self.room_group_name,
        #     self.channel_name
        # )
        await self.accept()

    # async def disconnect(self, close_code):
    #     # Leave the room group
    #     await self.channel_layer.group_discard(
    #         self.room_group_name,
    #         self.channel_name
    #     )

    async def receive(self, text_data):
        # Receive audio data from WebSocket
        data = json.loads(text_data)
        audio_data = data.get('audio_data')
        print("audio_data:", audio_data)

        # Process audio data (you can apply your own analysis here)
        audio_waveform = self.process_audio(audio_data)

        # Send the plot back to the frontend
        await self.send(text_data=json.dumps({
            'plot': audio_waveform
        }))

    def process_audio(self, audio_data):
        # Convert audio data to numpy array (assuming it's a list of audio samples)
        audio_samples = np.frombuffer(audio_data, dtype=np.int16)

        # Create a plot of the sound wave
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(audio_samples)
        ax.set_title("Sound Wave")
        ax.set_xlabel("Sample Number")
        ax.set_ylabel("Amplitude")

        # Save the plot to a BytesIO buffer and encode it as base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)

        return plot_data