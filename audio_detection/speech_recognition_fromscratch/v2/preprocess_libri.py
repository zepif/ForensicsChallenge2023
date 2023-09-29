import os
import torch
import torchaudio

DATASET = "C:\\fun\ForensicsChallenge2023_team6\\audio_detection\speech_recognition_fromscratch\LibriSpeech"

SAMPLE_RATE = 1600
def load_example(x):
    waveform, sample_rate = torchaudio.load(x, normalize=True)
    if sample_rate != SAMPLE_RATE:
        print(f"Sample rate mismatch in file {x}. Expected {SAMPLE_RATE}, but got {sample_rate}.")
    assert(sample_rate == SAMPLE_RATE)
    mel_specgram = mel_transform(waveform)
    return mel_specgram[0].T

if __name__ == '___main__':
    for d in os.listdir(os.path.join(DATASET, 'train-clean-100')):
        for dl in os.listdir(os.path.join(DATASET, 'train-clean-100', d)):
            for dll in os.listdir(os.path.join(DATASET, 'train-clean-100', d, dl)):
                fn = os.path.join(DATASET, 'train-clean-100', d, dl, dll)
                if fn.endswith(".flac"):
                    ret = torchaudio.load(fn)
                    print(ret)
                    exit(0)