import os
import csv
import torch
from tqdm.auto import tqdm
from torch import nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset

DATASET = "C:\\fun\ForensicsChallenge2023_team6\\audio_detection\speech_recognition_fromscratch\data"
CHARSET = " abcdefghijklmnopqrstuvwxyz,."
XMAX = 810
YMAX = 150

def get_metadata():
    ret = []
    with open(os.path.join(DATASET, 'metadata.csv'), newline = '') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        for row in reader:
            answer = [CHARSET.index(c)+1 for c in row[1].lower() if c in CHARSET]
            if len(answer) <= YMAX:
                ret.append((os.path.join(DATASET, 'wavs', row[0]+".wav"), answer))
    print("got metadata", len(ret))
    return ret

class LJSpeech(Dataset):
    def __init__(self):
        self.meta = get_metadata()
        self.sample_rate = 22050
        self.transform = torchaudio.transforms.MelSpectrogram(self.sample_rate, n_fft=1024, win_length=1024)

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        x, y = self.meta[idx]
        waveform, sample_rate = torchaudio.load(x, normalize=True)
        assert(sample_rate == self.sample_rate)
        mel_specgram = self.transform(waveform)
        #return 10+torch.log10(mel_specgram[0]).T, y
        return mel_specgram[0], y
    
class Rec(nn.Module):
    def __init__(self):
        super().__init__()
        self.prepare = nn.Sequential(
            nn.Linear(80, 120),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.encoder = nn.GRU(128, 128, batch_first=False)
        self.decode = nn.Linear(128, len(CHARSET))
    
    def forward(self, x):
        x = self.prepare(x)
        x = nn.fuctional_relu(self.encoder(x)[0])
        x = self.decode(x)
        return torch.nn.functional.log_softmax(x, dim=2)
    
def pad_sequence(batch):
    sorted_batch = sorted(batch, key = lambda x: x[0].shape[0], reverse=True)
    input_lengths = [x[0].shape[0] for x in sorted_batch]
    target_lengths = [len(x[1]) for x in sorted_batch]
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False)
    labels = [x[1] + [0]*[YMAX - len(x[1])] for x in sorted_batch]
    labels = torch.LongTensor(labels)
    return sequences_padded, labels[:, :max(target_lengths)], input_lengths, target_lengths

def get_dataloader(batch_size):
    dset = LJSpeech()
    trainloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=8, 
                                                collate_fn=None)
    return dset, trainloader

def train(): 
    batch_size=32
    dset, trainloader = get_dataloader(batch_size)
    #ctc_loss = nn.CTCLoss(reduction='mean', zero_infinity='True').cuda()
    #model = Rec().cuda()
    ctc_loss = nn.CTCLoss(reduction='mean', zero_infinity='True')
    model = Rec()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        t = tqdm(trainloader, total=len(dset)//batch_size)
        for data in t:
            input, target, input_length, target_length = data
            #input = input.cuda()
            #target = target.cuda()
            optimizer.zero_grad()
            guess = model(input)
            pp = ''.join([CHARSET(c-1) for c in guess[:, 0, :].argmax(dim=1).cpu() if c != 0])
            if len(pp) > 0:
                print(pp)
            loss = ctc_loss(guess, target, input_length, target_length)
            loss.backward()
            optimizer.step()
            t.set_description("loss: %.2f" % loss.item())

if __name__ == '__main__':
    train()