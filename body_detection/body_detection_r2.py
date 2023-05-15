import cv2
import os

MODEL_DIR = os.path.join('D:', os.sep, 'ai_sweden_2023','body_detection')
model1_path = os.path.join(MODEL_DIR, 'haarcascade_frontalface_default.xml')
model2_path = os.path.join(MODEL_DIR, 'haarcascade_profileface.xml')
model3_path = os.path.join(MODEL_DIR, 'haarcascade_fullbody.xml')

face_cascade = cv2.CascadeClassifier(model1_path)
face_cascade_2 = cv2.CascadeClassifier(model2_path)
body_cascade = cv2.CascadeClassifier(model3_path)

VIDEOS_DIR = os.path.join('D:', os.sep, 'ai_sweden_2023','body_detection')
video_path = os.path.join(VIDEOS_DIR, 'video.mp4')
video_path_out = 'video_out22.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

while True:
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.15, minNeighbors = 3)
    faces_2 = face_cascade.detectMultiScale(gray, scaleFactor = 1.15, minNeighbors = 3)
    bodies = body_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 3)

    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
      cv2.putText(frame, 'face', (x,y), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,255,255), 1)

    for (x, y, w, h) in faces_2:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
      cv2.putText(frame, 'face', (x,y), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,255,255), 1)

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'person', (x,y), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,255,255), 1)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
