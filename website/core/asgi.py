"""
ASGI config for core project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/asgi/
"""

import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
import heed.routing


from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")


application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websockets": AuthMiddlewareStack(
            URLRouter(heed.routing.websocket_urlpatterns)
        ),
    }
)

# application = get_asgi_application()
