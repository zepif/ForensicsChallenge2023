import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade_2 = cv2.CascadeClassifier('haarcascade_profileface.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

cap = cv2.VideoCapture('video.mp4')

while True:
    ret, img = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.15, minNeighbors = 3)
    faces_2 = face_cascade.detectMultiScale(gray, scaleFactor = 1.15, minNeighbors = 3)
    bodies = body_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 3)

    for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for (x, y, w, h) in faces_2:
      cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for (x, y, w, h) in bodies:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
