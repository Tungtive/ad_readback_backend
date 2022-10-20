import base64
import hashlib
from typing import Dict

# import bcrypt
from flask_mongoengine import Document
from mongoengine.fields import StringField

# def get_hashed_password(plain_text_password):
#     return bcrypt.hashpw(
#         base64.b64encode(hashlib.sha256(plain_text_password.encode("utf-8")).digest()),
#         bcrypt.gensalt(),
#     ).decode("utf8")


# def check_password(plain_text_password, hashed_password):
#     return bcrypt.checkpw(
#         base64.b64encode(hashlib.sha256(plain_text_password.encode("utf-8")).digest()),
#         hashed_password.encode("utf8"),
#     )


class User(Document):

    username = StringField(required=True, unique=True, max_length=36)
    password = StringField(required=True)
    phone_number = StringField()
    email  = StringField()


    def to_dict(self):
        return  {
            "id" :str(self.id),
            "username":self.username,
            "password":self.password,
            "phone_number":self.phone_number,
            "email":self.email
        }


    # def __init__(self,user_id,user_name,user_mobile_number,user_email_address) -> 'User':
    #     self.id = user_id
    #     self.name = user_name
    #     self.mobile_number = user_mobile_number
    #     self.email_address  =  user_email_address
    # def to_dict(self) -> Dict:
    #     return{
    #         "id": self.id,
    #         "name": self.name,
    #         "mobile_number":self.mobile_number,
    #         "email_address":self.email_address
    #     }