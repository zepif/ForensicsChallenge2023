import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

def create_model():
    num_rows = 4
    num_columns = 10
    num_channels = 1
    filter_size = 2
    num_labels = 2
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding="same", input_shape=(num_rows, num_columns, num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=1))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=1))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=1))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=2, padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=1))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_labels, activation='sigmoid'))
    return model

def load_trained_model():
    model = create_model()
    WEIGHTS_DIR = os.path.join('D:', os.sep, 'ai_sweden_2023', 'audio_detection')
    weights_path = os.path.join(WEIGHTS_DIR, 'gunshot_detection_weights.hdf5')
    model.load_weights(weights_path)
    return model

def gunshot_detection(audio_file_path, threshold):
    
    model = load_trained_model()

    data, sr = librosa.load(audio_file_path, sr=22050, mono=True)

    
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmax=8000)

    
    S = librosa.power_to_db(S, ref=np.max)
    S = np.expand_dims(S, axis=-1)
    S = np.expand_dims(S, axis=0)

    
    predicted = model.predict(S)
    
    
    shot_times = np.where(predicted[0,:,1] >= threshold)[0]
    shot_times = librosa.frames_to_time(shot_times, sr=sr)

    return shot_times

AUDIO_DIR = os.path.join('D:', os.sep, 'ai_sweden_2023', 'audio_detection')
audio_path = os.path.join(AUDIO_DIR, 'audio.wav')
shot_times = gunshot_detection(audio_path, 0.5)
print('Shot times: ', shot_times)
