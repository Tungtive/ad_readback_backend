import datetime
from sys import stdout
from xml.dom import UserDataHandler

from flask import jsonify, request
# from flask_jwt_extended import create_access_token, current_user, jwt_required
from flask_restx import Namespace, Resource

from .model import User

api = Namespace('users',description="users related api")




# user_db = [
#     User("1","Tom","18692077799","1194953762@qq.com"),
#     User("2","Jerry","0452663508","zunjiex@student.unimelb.edu.au")
# ]

@api.route("/")
class UserListApi(Resource):
    def get(self):
        # return jsonify([user.to_dict() for user in user_db])
        return [user.to_dict() for user in  User.objects()]


# def list_students():
#     if request.method == "GET":
#         return jsonify([student.to_dict() for student in student_db])
#     if request.method == "POST":
     
@api.route("/register")
class  UserRegisterApi(Resource):
    def post(self):
        data =  request.json
        user = User()
        username  = data['username']
        for user1 in User.objects():
            if (user1.to_dict()['username']==username ): 
                return {"error":"username already exist!"},401
        user.username = username
        user.password = data['password']
        user.phone_number = data['phone_number']
        user.email = data['email']
        user.save()
        return {"user":user.to_dict(),"register":"ok"},201
# @api.route("/<user_id>")
# class UserApi(Resource):
#     def get(self, user_id):
#         # for user in user_db:
#         #     if user.id == user_id:
#         #         return user.to_dict()
#         # return "null", 404
#         return User.objects(id=user_id).first_or_404(message="User not found").to_dict()


auth_api = Namespace("auth")

@auth_api.route("/login")
class Login(Resource):
    def post(self):
        username = request.json.get("username")
        password = request.json.get("password")
        print(request.json)
        # if not username or not password:
        #     return {"error": "username or password is missing"}, 400
    
        user = User.objects(username=username).first_or_404(message="User not found")
        # if not check_password(password, user.password):
        #     return {"error": "password is incorrect"}, 401
        if user.password != password :
            return {"error": "password is incorrect"}, 401
        # jwt_token = create_access_token(
        #     identity=user.name, expires_delta=datetime.timedelta(days=30)
        # )
        return {"user": user.to_dict(),"login":"ok"}, 201


