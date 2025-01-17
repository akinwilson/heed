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

def plot(audio_bytes=None, c=plt_colours[1]):
    fig, ax = plt.subplots()
    x = np.linspace(0, 2 * np.pi, 1000)
    y = [np.cos(_x) for _x in x]
    ax.plot(x, y, "*", color=c)
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
            print(f"file:{file}")
            print("file.file", file.file)
            import numpy as np 
            audio_samples = np.frombuffer(file.file, dtype=np.int16)

            print("file.__dict__",file.__dict__ )

    
    return JsonResponse({'image_uri':plot_bar()}) 
    
def index(request):
    print("hello")
    return render(request, "heed/index3.html") # , {"image_uri": plot()}

    # )  # HttpResponse("First response")


# def index(request):
#     return StreamingHttpResponse(iterate_plot())


# Create your views here.
