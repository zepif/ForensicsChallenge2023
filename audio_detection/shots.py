from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import os

#audio = AudioSegment.from_file('video2.mp4', format='wav')
VIDEOS_DIR = os.path.join('D:', os.sep, 'ai_sweden_2023', 'audio_detection')
video_path = os.path.join(VIDEOS_DIR, 'video.mp4')
clip = VideoFileClip(video_path)
length = clip.duration
audio = clip.audio
audio.write_audiofile('audio.wav')
audio = AudioSegment.from_file('audio.wav', format='wav')
samples = np.array(audio.get_array_of_samples())

threshold = 0.8 * np.max(np.abs(samples))


shot_times = []
for i in range(len(samples)):
    if np.abs(samples[i]) >= threshold:
        shot_times.append(float(i) / (audio.frame_rate * 2))

print('Shot times:', shot_times)

time = np.arange(len(samples)) / (float(audio.frame_rate) * 2)
plt.plot(time, samples)
for shot_time in shot_times:
    plt.axvline(x=shot_time, color='r', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()
