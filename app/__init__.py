from distutils.command.config import config
from flask import Blueprint, Config, Flask
from flask_cors import CORS
from flask_mongoengine import MongoEngine
from flask_restx import Api

from app.dataHistory.controller import dataHistory_api
from app.user.controller import api as user_api
from app.user.controller import auth_api
import sys 
sys.path.append("..") 
from config import Config
api_bp = Blueprint("api", __name__, url_prefix="/api/v1")
api = Api(api_bp)

api.add_namespace(user_api)
api.add_namespace(auth_api)
api.add_namespace(dataHistory_api)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    print(app.config)
    MongoEngine(app)
    CORS(app)
    app.register_blueprint(api_bp)
    return app