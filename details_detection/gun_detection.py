from roboflow import Roboflow
import os
import cv2
rf = Roboflow(api_key="1imvMdiEUwSHIDrKVJO5")
project = rf.workspace().project("gun-detection-s5poj")
model = project.version(1).model

VIDEOS_DIR = os.path.join('D:', os.sep, 'ai_sweden_2023','details_detection', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'video3.mp4')
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
video_path_out = 'video_out.mp4'.format(video_path)
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

while ret:
    predictions = model.predict(frame).json()['predictions']
    for prediction in predictions:
        x1 = prediction['x'] - prediction['width'] / 2
        x2 = prediction['x'] + prediction['width'] / 2
        y1 = prediction['y'] - prediction['height'] / 2
        y2 = prediction['y'] + prediction['height'] / 2
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(frame, 'GUN', (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        
    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
