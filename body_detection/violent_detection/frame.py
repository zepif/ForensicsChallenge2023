import os
import cv2
import torch

video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

frame_dir = 'frames'
os.makedirs(frame_dir, exist_ok=True)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_filename = f'{frame_dir}/frame_{frame_count}.jpg'
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

cap.release()

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

base_model = ResNet50(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)

feature_extractor = Model(inputs=base_model.input, outputs=x)

frames = []

for i in range(frame_count):
    frame_path = f'{frame_dir}/frame_{i}.jpg'
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    frames.append(img)

frames = np.array(frames)

features = feature_extractor.predict(frames)
