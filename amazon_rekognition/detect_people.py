import os
import boto3
import cv2
import credentials
import matplotlib.pyplot as plt

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

reko_client = boto3.client('rekognition', 
                           aws_access_key_id=credentials.access_key,
                           aws_secret_access_key=credentials.secret_key,
                           region_name='eu-central-1')

input_file = 'D:/ai_sweden_2023/amazon_rekognition/video2.mp4'
cap = cv2.VideoCapture(input_file)
rett, framee = cap.read()
Hh, Ww, _ = framee.shape

video_path_out = 'video_res2.mp4'.format(input_file)
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (Ww, Hh))
counter = 0

ret = True
class_names = ['Person']
while ret: 
    cap.set(cv2.CAP_PROP_POS_FRAMES, counter)
    ret, frame = cap.read()
    if ret:
        H, W, _ = frame.shape
        tmp_filename = 'D:/ai_sweden_2023/amazon_rekognition/tmp.jpg'
        cv2.imwrite(tmp_filename, frame)

        with open(tmp_filename, 'rb') as image:
                response = reko_client.detect_labels(Image={'Bytes': image.read()})

        for label in response['Labels']:
            if len(label['Instances']) > 0:
                name = label['Name']
                if name in class_names:
                    for instance in label['Instances']:
                        conf = float(instance['Confidence']) / 100
                        w = instance['BoundingBox']['Width']
                        h = instance['BoundingBox']['Height']
                        x = instance['BoundingBox']['Left']
                        y = instance['BoundingBox']['Top']
                        x_ = int(x * W)
                        w_ = int(w * W)
                        y_ = int(y * H)
                        h_ = int(h * H)
                        frame = cv2.rectangle(frame, (x_, y_), (x_ + w_, y_ + h_), (0, 255, 0), 2)

        out.write(frame)
        os.remove(tmp_filename)
        counter += 1

cap.release()
out.release()
cv2.destroyAllWindows()
