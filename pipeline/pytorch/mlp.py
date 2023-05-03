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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class TrimerMLPDataset(Dataset):
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column

    def __getitem__(self, index):
        tokens = self.data.loc[index, "fps"]
        target = self.data.loc[index, self.target_column]

        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return tokens_tensor, target_tensor

    def __len__(self):
        return len(self.data)
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        # First hidden layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        init.xavier_uniform_(self.layers[-1].weight)
        init.zeros_(self.layers[-1].bias)

        # Additional hidden layers
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            init.xavier_uniform_(self.layers[-1].weight)
            init.zeros_(self.layers[-1].bias)

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        init.xavier_uniform_(self.layers[-1].weight)
        init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = nn.functional.relu(self.layers[i](x))
            #print(f"Output shape at hidden layer {i + 1}: {x.shape}")

        x = self.layers[-1](x)
        x = torch.squeeze(x, dim=2)
        return x

    def __str__(self):
        layer_str = '\n'.join([f'Layer {i}: {layer}' for i, layer in enumerate(self.layers)])
        return f'MLP Model:\n{layer_str}'

    
def train_MLP(model, train_loader, test_loader, loss_fn, optimizer, scheduler, num_epochs=10, device="cpu", wandb_log = False):
    model.to(device)
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for x, y in train_loader:
            #x, y = x.to(device), y.to(device)
            x = x.to(device).float()
            y = y.to(device).float()
            y = y.view(-1, 1)

            optimizer.zero_grad()
            output = model(x)
            #output = torch.squeeze(output, dim=2)
            
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Testing
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                #x, y = x.to(device), y.to(device)
                x = x.to(device).float()
                y = y.to(device).float()
                y = y.view(-1, 1)

                output = model(x)
                #output = torch.squeeze(output, dim=2)
                
                loss = loss_fn(output, y)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        if wandb_log:
            wandb.log({"train_loss": test_loss, "val_loss": test_loss})

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        #### Update the learning rate scheduler
        if scheduler != None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

    return train_losses, test_losses