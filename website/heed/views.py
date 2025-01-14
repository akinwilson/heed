from django.shortcuts import render
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
    print(request)
    if request.POST:
        print(request)

def index(request):
    image_url = "https://wcs.smartdraw.com/chart/img/basic-bar-graph.png?bn=15100111801"
    return render(
        request, "heed/index.html", {"image_uri": plot()}
    )  # HttpResponse("First response")


# def index(request):
#     return StreamingHttpResponse(iterate_plot())


# Create your views here.
