import json
import os
import datetime
import requests
from pprint import pprint
folder=os.getcwd()
modify='modify'
for path,dirs,files in os.walk('./folder'):
    for fname in files:
        with open(fname,encoding='utf-8') as data_file:
            data=json.loads(data_file.read())
            print_data=data
            print_data["people"][0]["face_keypoints"]=data["people"][0]["face_keypoints_2d"]
            print_data["people"][0]["hand_left_keypoints"]=data["people"][0]["hand_left_keypoints_2d"]
            print_data["people"][0]["hand_right_keypoints"]=data["people"][0]["hand_right_keypoints_2d"]
            print_data["people"][0]["pose_keypoints"]=data["people"][0]["pose_keypoints_2d"]

            del print_data["people"][0]["face_keypoints_2d"]
            del print_data["people"][0]["face_keypoints_3d"]
            del print_data["people"][0]["hand_left_keypoints_2d"]
            del print_data["people"][0]["hand_left_keypoints_3d"]
            del print_data["people"][0]["hand_right_keypoints_2d"]
            del print_data["people"][0]["hand_right_keypoints_3d"]
            del print_data["people"][0]["pose_keypoints_2d"]
            del print_data["people"][0]["pose_keypoints_3d"]

            with open(modify+'\\'+fname,"w",encoding='utf-8') as write_file:
                json.dump(print_data,write_file,ensure_ascii=False)


