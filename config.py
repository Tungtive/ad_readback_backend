import os
from typing import Literal

from click import MultiCommand
from flask import Config


class Config(Config):
    enable_utc = True
    APP_NAME : Literal['app'] = "app"
    MONGODB_DB : str = os.getenv("MONGODB_DB","api")
    MONGODB_USERNAME : str = os.getenv("MONGODB_USERNAME","root")
    MONGODB_PASSWORD : str = os.getenv("MONGODB_PASSWORD","root")
    MONGODB_HOST : str = os.getenv("MONGODB_HOST","localhost")
    MONGODB_PORT : Literal[27017] = 27017

# class Config(Config):
#     enable_utc = True
#     APP_NAME = "app"
#     MONGODB_DB = "api"
#     MONGODB_USERNAME = "root"
#     MONGODB_PASSWORD = "root"
#     MONGODB_HOST = "127.0.0.1"
#     MONGODB_PORT = 27017
#     # JWT_SECRET_KEY = "asjkdfhlasdjfkljaklsdjflkjasd;lf"
#     # ERROR_404_HELP = False
#     # AWS_BUCKET_NAME = "sx-spa-demo"