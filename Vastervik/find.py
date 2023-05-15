import cv2
import pandas as pd
import os
import numpy as np
from ultralytics import YOLO


model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

PICTURES_DIR = os.path.join('D:', os.sep, 'ai_sweden_2023','Vastervik')
pictures_path = os.path.join(PICTURES_DIR, 'picture3.jpg')
pictures_path_out = 'picture3_out.jpg'.format(pictures_path)


COCO_DIR = os.path.join('D:', os.sep, 'ai_sweden_2023','Vastervik')
coco_path = os.path.join(COCO_DIR, 'coco.txt')
my_file = open(coco_path, "r")
data = my_file.read()
class_list = data.split("\n") 

results = model(pictures_path, save=True)