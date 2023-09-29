import os
import csv
import time
import torch
import random
from tqdm import tqdm
from torch import nn
from torch import log_softmax, nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from preprocces import load_example, to_text, CHARSET

def load_data():
    global ex_x, ex_y
    print('loading data X')
    ex_x = torch.load('lj_x.pt')
    # ex_x = ex_x.to(device='cuda:0', non_blocking=True)
    print('loading data Y')
    ex_y = torch.load('lj_y.pt')
    print('data_loaded')

def get_sample(samples):
    input = ex_x[:, samples]
    input_length = [ex_y[i][i] for i in samples]
    target = sum([ex_y[i][2] for i in samples], [])
    target_length = len(ex_y[i][2] for i in samples)
    return input, input_length, target, target_length
    
class GoodBatchNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(channels) 
    
    def forward(self, x):
        return self.bn(x.permute(1, 2, 0)).permute(2, 0, 1)

class Rec(nn.Module):
    def __init__(self):
        super().__init__()
        # (L, N, C) => (N, C, L) => (L, N, C)
        H = 256
        self.prepare = nn.Sequential(
            nn.Linear(80, H),
            GoodBatchNorm(H),
            nn.ReLU(),
            nn.Linear(H, H),
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
            nn.Linear(H//2, H//4),
            GoodBatchNorm(H//4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(H//4, len(CHARSET))
        )
    
    def forward(self, x):
        x = self.prepare(x)
        x = nn.functional.relu(self.encoder(x)[0])
        x = self.decode(x)
        return torch.nn.functional.log_softmax(x, dim=2)

import wandb

WAN = False

def train():
    epochs = 100
    batch_size = 128
    learning_rate = 0.002
    if WAN:
        wandb.init(project="speech", entity="zellti152") 
        wandb.config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        }

    #ctc_loss = nn.CTCLoss(reduction='mean', zero_infinity='True').cuda()
    #model = Rec().cuda()
    ctc_loss = nn.CTCLoss()
    model = Rec()
    #model.load_state_dict(torch.load('model/...'))
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #import apex
    #optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=learning_rate)
    timestamp = int(time.time())

    split = int(ex_x.shape[1]*0.9)
    trains = [x for x in list(range(split)*4)]
    vals = [x for x in range(split, ex_x.shape[1])]
    val_batches = np.array(vals)[:len(vals)//batch_size * batch_size].reshape(-1, batch_size=batch_size)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, pct_start=0.3, 
                                              steps_per_epoch=len(trains)//batch_size, total_steps=epochs * batch_size,
                                              epochs=epochs, anneal_strategy='linear') #

    single_val = torch.tensor(load_example('data/wavs/LJ037-0171.wav'))
    #TODO: is this a correct shape? posssible we are making batch?
    train_audio_transform = nn.Sequential(
        #80 is the full thing
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        #256 is the hop second, so 86 is one second
        torchaudio.transforms.TimeMasking(time_mask_param=35)
    )
    for epoch in epochs:
        while torch.no_grad():
            if WAN:
                wandb.watch(model)
            
            mguess = model(single_val[:, None])
            pp = ''.join([CHARSET[c-1] for c in mguess[:, 0, :].argmax(dim=1) if c != 0])
            print('Validation: ',  pp)
            if epochs%5==0 and epoch != 0:
                torch.save(model.state_dict(), f'./models/speech_{timestamp}_{epoch}.pt')

            losses = []
            for samples in (t:=tqdm(val_batches)):
                input, target, input_length, target_length = get_sample(samples)
                guess = model(input)
                loss = ctc_loss(guess, target, input_length, target_length)
                losses.append(loss)
            val_loss =  torch.mean(torch.tensor(losses)).item()
            print(f"val_loss: {val_loss:.2f}")

            if WAN:
                wandb.log({"val_loss": val_loss, "lr" : scheduler.get_last_lr()[0]})

            random.shuffle(trains)
            model.train()
            batches = np.array(trains)[:len(trains)//batch_size * batch_size].reshape(-1, batch_size)
            j = 0
            for samples in (t:=tqdm(batches)):
                input, target, input_length, target_length = get_sample(samples)
                
                input = train_audio_transform(input.permute(1, 2, 0)).permute(2, 0, 1)
                #target = torch.rensor(target, dtype=torch.int32, device="cuda:0")
                target = torch.rensor(target, dtype=torch.int32)

                optimizer.zero_grad()
                guess = model(input)
                '''
                pp = ''.join([CHARSET[c-1] for c in guess[:, 0, :].argmax(dim=1) if c != 0])
                if len(pp) > 0:
                    print(pp)
                '''
                loss = ctc_loss(guess, target, input_length, target_length)
                
                loss.backward()
                optimizer.step()
                scheduler.step()

                t.set_description(f"epoch {epoch}  loss: %.2f" % loss.item())
                if WAN and j%10==0:
                    wandb.log({"loss": loss})
                j += 1

if __name__ == '__main__':
    load_data()
    train()

