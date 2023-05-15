from ultralytics import YOLO
 
model = YOLO('yolov8s_gun.yaml')

results = model.train(data='config.yaml', epochs=25)
