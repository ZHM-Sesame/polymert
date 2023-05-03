import pandas as pd
import numpy as np
import pickle
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import time
import argparse
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.init as init
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm

class MolecularRNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y

# adjusted from Veronika
class RNN(nn.Module):
    def __init__(self, seq_len, emb_dim, lstm_dim, linear_dim, out_dim, num_tokens=None):
        super(RNN, self).__init__()
        self.seq_len = seq_len
        self.emb = nn.Embedding(num_tokens, emb_dim) if num_tokens is not None else None
        self.lstm1 = nn.LSTM(emb_dim, lstm_dim, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_dim*2, lstm_dim, bidirectional=True, batch_first=True)
        self.lstm_dim = lstm_dim
        
        # Apply Glorot uniform initialization to LSTM layers
        for name, param in self.lstm1.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.zeros_(param)
        for name, param in self.lstm2.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.zeros_(param)
        
        self.mlp = nn.Sequential(
            nn.Linear(lstm_dim * 2, int(lstm_dim/2)),
            #nn.Linear(lstm_dim * 2, int(lstm_dim)),
            nn.ReLU(),
        )
        init.xavier_uniform_(self.mlp[0].weight)
        init.zeros_(self.mlp[0].bias)

        self.last = nn.Linear(int(lstm_dim/2)*seq_len, out_dim)
        #self.last = nn.Linear(int(lstm_dim/2), out_dim)

        init.xavier_uniform_(self.last.weight)
        init.zeros_(self.last.bias)


    def forward(self, data):  # 2D
        #x = self.emb(data) if self.emb else data
        #print(x.shape)
        x = data
        x, _ = self.lstm1(x)
        #print(f"LSTM Layer 1 output shape: {x.shape}")

        x, _ = self.lstm2(x)
        #print(f"LSTM Layer 2 output shape: {x.shape}")
        
        x = self.mlp(x)

        #x = x.reshape(x.shape[0], -1)  # Flatten time dim into last one (instead of before into sample dim)
        x = x.reshape(-1, int(self.lstm_dim/2) * self.seq_len)
        
        #print(f"Reshape output shape: {x.shape}")

        x = self.last(x)
        #print(f"Output Layer shape: {x.shape}")

        return x
    
    def __str__(self):
        layers = [
            ('Embedding Layer', self.emb),
            ('LSTM Layer 1', self.lstm1),
            ('LSTM Layer 2', self.lstm2),
            ('MLP', self.mlp),
            ('Output Layer', self.last),
        ]
        layer_str = '\n'.join([f'{name}: {layer}' for name, layer in layers])
        return f'RNN Model:\n{layer_str}'

    
def train_RNN(model, train_loader, test_loader, loss_fn, optimizer, scheduler, num_epochs=10, device="cpu", wandb_log = False):
    model.to(device)
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        #for x, y in train_loader:
        step = 0
        losses = []
        ys = []
        for x, y in tqdm(train_loader, desc=f'Training epoch {epoch + 1}/{num_epochs}'):
            x, y = x.to(device), y.to(device)
            y = y.view(-1, 1)
            ys.append(y)
            
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            losses.append(loss.item())
            step+=1
        
        loader_len = len(train_loader)
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Testing
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y = y.view(-1, 1)

                output = model(x)
                loss = loss_fn(output, y)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        if wandb_log:
            wandb.log({"train_loss": test_loss, "val_loss": test_loss})

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {test_loss:.4f}")
        
        # Update the learning rate scheduler
        if scheduler != None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

    return train_losses, test_losses