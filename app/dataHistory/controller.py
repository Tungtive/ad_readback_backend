
import datetime
from unittest import result

from flask import jsonify, request
from flask_restx import Namespace, Resource

from .model import DataHistory

dataHistory_api = Namespace('dataHistory',description="dataHistory related api")

@dataHistory_api.route("/")
class UserListApi(Resource):
    def get(self):
        # return jsonify([user.to_dict() for user in user_db])
        return [dataHistory.to_dict() for dataHistory in  DataHistory.objects()]

     
@dataHistory_api.route("/getDataHistoryByName")
class  GetDataHistroyByUsernameApi(Resource):
    def post(self):
        name = request.json.get("name")
        for dataHistory in DataHistory.objects():
            if dataHistory.to_dict()['name']==name:
                 x = dataHistory.to_dict()['angleData_x']
                 y = dataHistory.to_dict()['angleData_y']
                 z = dataHistory.to_dict()['angleData_z']
                 return {"x": x,"y":y,"z":z}, 201
        return {"error":"record not found"}


@dataHistory_api.route("/getDataHistoryByUsername")
class  GetDataHistroyBynameApi(Resource):
    def post(self):

        username = request.json.get("username")
        result = []
        for datahistory in DataHistory.objects():
            if datahistory.to_dict()['username']==username:
                result.append(datahistory.to_dict()["name"])
        if len(result)== 0:
            return {"error":"empty history data"}
        return result

@dataHistory_api.route("/uploadDataHistory")
class  UploadDataHistroyApi(Resource):
    def post(self):
        data =  request.json
        dataHistory = DataHistory()
        dataHistory.username = data['username']
        count =  0;
        for index in DataHistory.objects():
            if  index.username == dataHistory.username:
                count += 1;
        dataHistory.name = dataHistory.username + '_data_history_' + str(count)
        dataHistory.angleData_x = data['angledata_x']
        dataHistory.angleData_y = data['angledata_y']
        dataHistory.angleData_z = data['angledata_z']
        dataHistory.save()
        return {"dataHistory":dataHistory.to_dict(),"upload":"ok"},201

