import json
import resource
from calendar import c
from datetime import datetime

from dotenv import load_dotenv
from flask import Blueprint, Flask
from flask_mongoengine import MongoEngine
from flask_restx import Api, Resource

from app.dataHistory.controller import dataHistory_api
from app.user.controller import api as user_api
from app.user.controller import auth_api
from config import Config

load_dotenv()
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

api_bp = Blueprint("api", __name__, url_prefix="")
api = Api(app)
api.add_namespace(user_api)
api.add_namespace(auth_api)
api.add_namespace(dataHistory_api)

if __name__ == "__main__":
    print(app.config['MONGODB_HOST'])
    app.run(host="0.0.0.0",port=3005,debug=True)

