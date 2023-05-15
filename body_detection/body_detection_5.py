import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO


model = YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('video.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count = 0

while True:    
    ret, frame = cap.read()

    if not ret:
        break

    count += 1

    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020,500))
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


    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
