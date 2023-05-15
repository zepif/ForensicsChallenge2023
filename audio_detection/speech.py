import os
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from nltk.tokenize import word_tokenize, sent_tokenize

VIDEOS_DIR = os.path.join('D:', os.sep, 'ai_sweden_2023', 'creation_video')
video_path = os.path.join(VIDEOS_DIR, 'video2.mp4')
clip = VideoFileClip(video_path)
audio = clip.audio
audio.write_audiofile('audio.wav')

r = sr.Recognizer()
with sr.AudioFile('audio.wav') as source:
    audio_text = r.record(source)

text = r.recognize_google(audio_text, language='en-US')

file_name = "transcription.txt"

with open(file_name, "w") as file:
    file.write(text)

os.system(f"start {file_name}")