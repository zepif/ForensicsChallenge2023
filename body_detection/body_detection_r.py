import os
import pandas as pd
from ultralytics import YOLO
import cv2


#VIDEOS_DIR = os.path.join('.', 'videos')
VIDEOS_DIR = os.path.join('D:', os.sep, 'ai_sweden_2023','body_detection')
video_path = os.path.join(VIDEOS_DIR, 'video.mp4')
video_path_out = 'video_out21.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

#model_path = os.path.join('.', 'models', 'last.pt')
model_path = os.path.join('D:', os.sep, 'ai_sweden_2023','body_detection', 'yolov8s.pt')
# Load a model
model = YOLO(model_path)  # load a custom model

COCO_DIR = os.path.join('D:', os.sep, 'ai_sweden_2023','body_detection')
coco_path = os.path.join(COCO_DIR, 'coco.txt')
my_file = open(coco_path, "r")
data = my_file.read()
class_list = data.split("\n") 

while ret:
    results = model.predict(frame)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")

    list = []
             
    for index,row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        #print(c)
        if 'person' in c:
           cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
           cv2.putText(frame, str(c), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,255,255), 1)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()