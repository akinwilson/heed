from django.urls import re_path

from . import consumers

websockets_urlpattern = [
    re_path(r"listen", consumers.KWSConsumer.as_asgi()),
]
