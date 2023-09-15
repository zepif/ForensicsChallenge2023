import cv2
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sentencepiece as spm  # Import SentencePiece
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util

# Initialize GPT-2
model_name = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def detect_objects(frame):
    # Perform object detection on the frame
    detections = object_detector.detect_objects(frame)
    
    object_labels = []

    for detection in detections:
        object_labels.append(detection['class'])

    return object_labels

def generate_formal_video_description(video_path, video_title):
    cap = cv2.VideoCapture(video_path)
    
    description = "This video, titled '" + video_title + "', contains the following scenes:\n\n"
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Perform object detection on the frame and get labels
        object_labels = detect_objects(frame)
        
        text = "In this scene, we see "
        
        if object_labels:
            text += ", ".join(object_labels) + "."
        else:
            text += "unidentified objects."

        description += text + "\n\n"
    
    cap.release()
    
    # Generate text with T5 based on the description
    input_text = "Please describe the content of this video in a formal and detailed manner:\n\n" + description
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2, num_beams=4, early_stopping=True)
    generated_description = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_description

if __name__ == "__main__":
    video_path = "../body_detection/video.mp4"
    video_title = "Video of a Crime"  # Updated video title
    description = generate_formal_video_description(video_path, video_title)
    print(description)
