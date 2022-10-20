from datetime import datetime
from typing import Dict

from flask_mongoengine import Document
from mongoengine import *
from mongoengine.fields import DateField, ListField, StringField
from pkg_resources import require

# from typing_extensions import Required



class DataHistory(Document):
    name = StringField(required=True,max_length=36)
    username = StringField(max_length=36,unique=False)
    dateTime = DateField(default=datetime.now)
    angleData_x = ListField(FloatField())
    angleData_y = ListField(FloatField())
    angleData_z = ListField(FloatField())


    def to_dict(self):
        return  {
            "id" :str(self.id),
            "name":self.name,
            "username" : self.username,
            "dateTime":self.dateTime.isoformat(),
            "angleData_x":self.angleData_x,
            "angleData_y":self.angleData_y,
            "angleData_z":self.angleData_z
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