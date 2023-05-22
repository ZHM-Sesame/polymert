from utils import load_dataset, load_embeddings, get_repeated_polymer, MolecularDataset, evaluate_model
from rnn import MolecularRNNDataset, RNN, train_RNN
from mlp import TrimerMLPDataset, MLP, train_MLP

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
from tqdm import tqdm
import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

def train(args):
    
    model_choice = args.model
    data_path = args.data
    embedding_path = args.embedding
    dataset_type = args.dataset
    repetitions = args.repetitions
    repetition_format = args.rep_style
    
    flag = False
    if dataset_type == "homopolymers":
        flag = True
    
    nbits, embeddings = load_embeddings(embedding_path)
    df = load_dataset(data_path, dataset_type)
    
    prop = args.prop
    if prop == 'ip':
        if dataset_type == 'copolymers':
            prop = "IP" #"Ei"
        elif dataset_type == "homopolymers":
            prop = "Ei"
        elif dataset_type == 'copolymers_2':
            prop = "IP_vs_SHE"
    elif prop == 'ea':
        if dataset_type == 'copolymers':
            prop = "EA" #"Eea"
        elif dataset_type == "homopolymers":
            prop = "Eea"
        elif dataset_type == 'copolymers_2':
            prop = "EA_vs_SHE"
    elif prop == "os":
        if dataset_type != 'copolymers_2':
            print("only copolymers_2 has oscillator_strength")
            return
        prop = "oscillator_strength"
    else:
        print("unknown property, please use either 'ip' or 'ea'.")
        return
        
    print("using the Model: "+ model_choice)
    print("using the property: "+ prop)
    df = df[df['property'] == prop]
    ncols = len(list(df.columns))
    
    
    
    nbits, Mix_X_100Block, target = get_repeated_polymer(df, embeddings, nbits, repetition_format, dataset_type, repetitions)
    
    X = Mix_X_100Block
    y = target
    
    # Split into temporary train set and final test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_temp.reset_index(drop=True, inplace=True)
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True, random_state=11)
    
    fold_no = 1
    for train, val in kfold.split(X_temp, y_temp):
        
        X_train, X_val = X_temp[train], X_temp[val]
        y_train, y_val = y_temp[train], y_temp[val]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        train_dataset = MolecularDataset(X_train_tensor, y_train_tensor)
        val_dataset = MolecularDataset(X_val_tensor, y_val_tensor)
        test_dataset = MolecularDataset(X_test_tensor, y_test_tensor)

        batch_size = 128
        shuffle = True
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

        num_epochs = 30
        if model_choice == 'RNN':
            seq_len = 100
            emb_dim = nbits
            lstm_dim = 20 #512
            linear_dim = 10 #256

            out_dim = 1
            learning_rate = 0.001
            weight_decay = 0.01 

            model = model = RNN(seq_len, emb_dim, lstm_dim, linear_dim, out_dim).to(device)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            train_losses, test_losses = train_RNN(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs, device)

        elif model_choice == 'MLP':
            input_dim = nbits
            hidden_dim = 1024
            num_layers = 9

            output_dim = 1
            learning_rate = 0.001

            model = MLP(input_dim, hidden_dim, output_dim, num_layers=num_layers).to(device)

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

            train_losses, test_losses = train_MLP(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, num_epochs, device)

        current_time = time.strftime("%Y%m%d-%H%M%S")
        model_path = f"models/model_{model_choice}_{prop}_{current_time}.pt"

        torch.save(model.state_dict(), model_path)
        print("Saved PyTorch Model State to "+model_path)

        # model evaluation
        # Evaluate the model
        predicted, targets = evaluate_model(model, test_dataloader, device)

        # Calculate R2 score, MAE, and RMSE
        r2 = r2_score(targets, predicted)
        mae = mean_absolute_error(targets, predicted)
        rmse = np.sqrt(mean_squared_error(targets, predicted))

        print(f"Fold numer: {fold_no}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        fold_no += 1
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default='RNN', help='Choose either "MLP" or"RNN"')
    parser.add_argument('--prop', type=str, required=False, default='ip', help='ip or ea')
    parser.add_argument('--data', type=str, required=False, default='../final_dataset/copolymers', help='data directory')
    parser.add_argument('--dataset', type=str, required=False, default='copolymers', help='=> "copolymers" or "homopolymers"')
    parser.add_argument('--embedding', type=str, required = False, default = '../pre_trained_embeddings/3dinfomax.csv', help='embedding type')
    #parser.add_argument('--embedding', type=str, required=False, default='../final_dataset/polymer_fingerprint_mappings/polymer_fingerprint_mappings.csv', help='embedding type')
    parser.add_argument('--repetitions', type=int, required=False, default=50, help='repetitions of the monomer unit')
    parser.add_argument('--rep_style', type=str, required=False, default='concat', help='weighted_add or concat')
    parsed_args = parser.parse_args()

    train(parsed_args)