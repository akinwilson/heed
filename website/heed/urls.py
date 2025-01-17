from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("upload_audio/", views.upload, name="upload_audio")
]
