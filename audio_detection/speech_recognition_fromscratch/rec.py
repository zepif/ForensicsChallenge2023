import os
import csv
import torch
from tqdm import tqdm
from torch import nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset

DATASET = "C:\\fun\ForensicsChallenge2023_team6\\audio_detection\speech_recognition_fromscratch\data"
CHARSET = " abcdefghijklmnopqrstuvwxyz,."
XMAX = 810
YMAX = 150
SAMPLE_RATE = 22050

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
    print("got metadata", len(ret))
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

cache = {}
def init_data():
    meta = get_metadata()
    for x, y in tqdm(meta):
        cache[x] = load_example(x), y
init_data()

import hashlib
class LJSpeech(Dataset):
    def __init__(self, val=False):
        self.meta = get_metadata()
        if val:
            cmp = lambda x: hashlib.sha1(x[0].encode('utf-8')).hexdigest()[0] == '0'
        else:
            cmp = lambda x: hashlib.sha1(x[0].encode('utf-8')).hexdigest()[0] != '0'
        self.meta = [x for x in self.meta if cmp(x)]
        print(f"set has {len(self.meta)}")
        #self.sample_rate = 22050

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        x, y = self.meta[idx]
        if x not in cache:
            print("never should happen")
            cache[idx] = load_example(x), y
        return cache[idx]
    
class GoodBatchNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        #return self.bn(x.permute(1, 2, 0)).permute(1, 2, 0)
        #return x
        xx = x.permute(1, 2, 0)
        xx = self.bn(xx)
        xx = xx.permute(2, 0, 1)
        return xx

class Rec(nn.Module):
    def __init__(self):
        super().__init__()
        # (L, N, C)
        H = 256
        self.prepare = nn.Sequential(
            nn.Linear(80, H),
            GoodBatchNorm(H),
            nn.ReLU(),
            nn.Linear(H, H),
            GoodBatchNorm(H),
            nn.ReLU()
        )
        #self.encoder = nn.GRU(H, H, batch_first=False, dropout=0.1)
        self.encoder = nn.GRU(H, H, batch_first=False)
        self.decode = nn.Sequential(
            nn.Linear(H, H//2),
            GoodBatchNorm(H//2),
            nn.ReLU(),
            nn.Linear(H//2, len(CHARSET))
        )
    
    def forward(self, x):
        x = self.prepare(x)
        x = nn.functional.relu(self.encoder(x)[0])
        x = self.decode(x)
        return torch.nn.functional.log_softmax(x, dim=2)
    
def pad_sequence(batch):
    sorted_batch = sorted(batch, key = lambda x: x[0].shape[0], reverse=True)
    input_lengths = [x[0].shape[0] for x in sorted_batch]
    target_lengths = [len(x[1]) for x in sorted_batch]
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False)
    # YMAX = max(input_length)
    #labels = [x[1] + [0]*(YMAX - len(x[1])) for x in sorted_batch]
    #labels = torch.LongTensor(labels)   
    #labels = labels[:, :max(target_lengths)]
    labels = sum([x[1] for x in sorted_batch], [])
    labels = torch.tensor(labels, dtype=torch.int32)
    return sequences_padded, labels, input_lengths, target_lengths

def get_dataloader(batch_size, val):
    dset = LJSpeech(val)
    trainloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True, 
                                                collate_fn=pad_sequence, pin_memory=True)
    return dset, trainloader

import wandb

WAN = False

def train():
    epochs = 300
    batch_size = 64
    learning_rate = 3e-4
    if WAN:
        wandb.init(project="speech", entity="zellti152") 
    config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    }
    dset, trainloader = get_dataloader(batch_size, False)
    valdset, valloader = get_dataloader(batch_size, True)
    #ctc_loss = nn.CTCLoss(reduction='mean', zero_infinity='True').cuda()
    #model = Rec().cuda()
    ctc_loss = nn.CTCLoss()
    model = Rec()
    #model.load_state_dict(torch.load('model/...'))
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #import apex
    #optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=learning_rate)
    val = torch.tensor(load_example('data/wavs/LJ037-0171.wav'))
    for epoch in range(epochs):
        if WAN:
            wandb.watch(model)
        
        if epoch % 2 == 0:
            mguess = model(val[:, None])
            pp = ''.join([CHARSET[c-1] for c in mguess[:, 0, :].argmax(dim=1) if c != 0])
            print('Validation: ',  pp)
            torch.save(model.state_dict(), f'./models/speech_{epoch}.pt')

        losses = []
        t = tqdm(valloader, total=len(valdset)//batch_size)
        for data in t:
            input, target, input_length, target_length = data
            guess = model(input)
            loss = ctc_loss(guess, target, input_length, target_length)
            losses.append(loss)
        val_loss =  torch.mean(torch.tensor(losses)).item()
        print(f"val_loss: {val_loss:.2f}")
        if WAN:
            wandb.log({"val_loss": val_loss})
 
        t = tqdm(trainloader, total=len(dset)//batch_size)
        for data in t:
            input, target, input_length, target_length = data
            input_length = torch.as_tensor(input_length)
            target_length = torch.as_tensor(target_length)

            #input = input.cuda()
            #input = input.to('cuda:0', non_blocking=True)
            #target = target.cuda()
            #target = target.to('cuda:0', non_blocking=True)
            optimizer.zero_grad()
            guess = model(input)
            '''
            pp = ''.join([CHARSET[c-1] for c in guess[:, 0, :].argmax(dim=1) if c != 0])
            if len(pp) > 0:
                print(pp)
            '''
            #print(input_length.type())
            #print(f'guess : {guess} ; target : {target} ; input size {input_length} ; target size : {target_length}')
            loss = ctc_loss(guess, target, input_length, target_length)
            
            loss.backward()
            optimizer.step()
            t.set_description("loss: %.2f" % loss.item())
            if WAN:
                wandb.log({"loss": loss})

if __name__ == '__main__':
    train()

