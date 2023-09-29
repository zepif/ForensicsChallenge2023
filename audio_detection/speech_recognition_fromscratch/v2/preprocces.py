import os
import functools
import torchaudio
import csv
from tqdm.auto import tqdm
import torch

DATASET = "C:\\fun\ForensicsChallenge2023_team6\\audio_detection\speech_recognition_fromscratch\data"
CHARSET = " abcdefghijklmnopqrstuvwxyz,."
SAMPLE_RATE = 22050
XMAX = 870
YMAX = 150

import itertools
def to_text(x):
    x = [x for x, g in itertools.groupby(x)]
    return ''.join([CHARSET[c - 1] for c in x if c != 0])

import functools
@functools.lru_cache(None)
def get_metadata():
    ret = []
    with open(os.path.join(DATASET, 'metadata.csv'), newline = '') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        for row in reader:
            answer = [CHARSET.index(c)+1 for c in row[1].lower() if c in CHARSET]
            if len(answer) <= YMAX:
                ret.append((os.path.join(DATASET, 'wavs', row[0]+".wav"), answer))
    return ret

mel_transform = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=1024, win_length=1024,
                                                              hop_length=1024, n_mels=80)
def load_example(x):
    waveform, sample_rate = torchaudio.load(x, normalize=True)
    if sample_rate != SAMPLE_RATE:
        print(f"Sample rate mismatch in file {x}. Expected {SAMPLE_RATE}, but got {sample_rate}.")
    assert(sample_rate == SAMPLE_RATE)
    mel_specgram = mel_transform(waveform)
    return mel_specgram[0].T

if __name__ == '___main__':
    meta = get_metadata()
    ex_x, ex_y = [], []
    for x, y in tqdm(meta):
        ex = load_example(x)
        ex_x.append(ex)
        ex_y.append((x, ex.shape[0], y))

    sequences_padded = torch.nn.utils.rnn.pad_sequence(ex_x, batch_first=False)
    print(sequences_padded.shape, sequences_padded.dtype)
    print(ex_y[0])
    torch.save(sequences_padded, 'date/lj_x.pt')
    torch.save(ex_y, 'date/lj_y.pt')