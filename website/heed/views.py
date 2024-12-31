from django.shortcuts import render
from django.http import HttpResponse


def index(request):
    return HttpResponse("First response")


# Create your views here.
