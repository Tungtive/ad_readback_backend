from calendar import c
from datetime import datetime
import resource

from flask import Flask
from flask_restx import Api, Resource
from apis.user.controller import api as user_api
from apis.user.controller import auth_api
from flask_mongoengine import MongoEngine

from config import Config

import json

# class CustomEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj,datetime.datetime):
#             return obj.isoformat()
#         elif isinstance(obj,ObjectId):
#             return str(obj)
#         else:
#             return super().default(obj)

app = Flask(__name__)

# app.config['MONGODB_SETTINGS'] = {
#     'db':'day1',
#     'host':'localhost',
#     'port': 27017

# }

app.config.from_object(Config)
# app.json_encoder = CustomEncoder
MongoEngine(app)

api = Api(app)
api.add_namespace(user_api)
api.add_namespace(auth_api)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=3002,debug=True)
