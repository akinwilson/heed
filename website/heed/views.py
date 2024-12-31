from django.shortcuts import render
from django.http import HttpResponse


def index(request):
    return render(request, "heed/index.html")  # HttpResponse("First response")


# Create your views here.
