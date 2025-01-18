import urllib.parse
from django.shortcuts import render
from django.http import JsonResponse
from django.http import StreamingHttpResponse
from django.template import loader
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import urllib
import random
import time
import numpy as np 
from pydub import AudioSegment
from io import BytesIO


plt_colours = ["b", "g", "r", "c", "m", "y"]


def plot_bar(prob=0.75):
    fig, ax = plt.subplots()
    x = ['Keyword', 'Not keyword']
    h = [prob, 1-prob]
    ax.bar(x,h)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    plot_encoded = base64.b64decode(buffer.read())
    uri = urllib.parse.quote(plot_encoded)
    return uri 

def plot(pcm=None, c=plt_colours[1]):
    fig, ax = plt.subplots()
    # x = np.linspace(0, 2 * np.pi, 1000)
    # y = [np.cos(_x) for _x in x]
    # ax.plot(x, y, "*", color=c)
    ax.plot(pcm,color=c)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    # print("plot() VIEWS: ", uri)
    return uri


# def iterate_plot():
#     for _ in range(10):
#         time.sleep(1)
#         t = loader.get_template("heed/index.html")
#         clr = random.sample(plt_colours, k=1)[0]
#         c = {"image_uri": plot(c=clr)}
#         yield t.render(c)


def upload(request):
    if request.method == "POST":
        if request.FILES.get("audio", False):
            file = request.FILES['audio']
            # print(f"file:{file}")
            # print("file.file", file.file)
            # import numpy as np 
            # print("dir(file.file)", dir(file.file))
            audio_bytes = BytesIO(b"".join(file.file.readlines()))
            # print("audio_file: ", audio_file)
            print("audio_file bytes: ", audio_bytes)


            audio = AudioSegment.from_ogg(audio_bytes) 
            audio.set_frame_rate(16000)

            # audio_file = audiosegment.from_file(audio_file, format='ogg')

            print("audio: ", audio)
            print("dir(audio): ", dir(audio))
            print("channel: ", audio.channels)

            pcm = np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (1 << (8 * audio.sample_width - 1))

            # print("array: ", audio.get_array_of_samples())
            # pcm = np.array(audio.get_array_of_samples())
            print("pcm.shape: ", pcm.shape)
            pcm_avg = np.mean(pcm, axis=1)
            # audio_samples = np.frombuffer(file.file, dtype=np.int16)
            print("pcm_avg.shape: ", pcm_avg.shape)
            # print("file.__dict__",file.__dict__ )

    return JsonResponse({'image_uri':plot(pcm=pcm_avg)}) 
    
def index(request):
    print("hello")
    return render(request, "heed/index3.html") # , {"image_uri": plot()}

    # )  # HttpResponse("First response")


# def index(request):
#     return StreamingHttpResponse(iterate_plot())


# Create your views here.
