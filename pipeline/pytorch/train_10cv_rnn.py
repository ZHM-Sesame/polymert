from utils import load_dataset, load_embeddings, get_repeated_polymer, MolecularDataset, evaluate_model
from rnn import MolecularRNNDataset, RNN, train_RNN

import pandas as pd
import numpy as np
import pickle
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
import subprocess
import gc
import csv

gpu_index = 0
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

# Save evaluation metrics to a unique CSV file
current_time = time.strftime("%Y%m%d-%H%M%S")
file_name = f"results/evaluation_metrics_{current_time}.csv"
file_name_review = f"results/evaluation_metrics_review_{current_time}.csv"

def train(embedding_choice, prop):
    
    ############ Initialize the hyperparameters #############
    num_epochs = 100
    model_choice = "RNN"
    data_path = "/home/stallam_umass_edu/new_changes/polymertl/final_dataset/copolymers"
    dataset_type = "copolymers_2"
    batch_size = 64
    test_portion = 0.2
    patience = 10
    
    if dataset_type in ["copolymers", "copolymers_2"]:
        lstm_dim = 128
        linear_dim = 32
        learning_rate = 0.001
        scheduler_choice = 'CosineAnnealingLR'
        repetitions = 100
        repetition_format = "concat"
    
    #set embedding path according to embedding choice
    embedding_dict = {
        'chemberta': "/home/stallam_umass_edu/new_changes/polymertl/pre_trained_embeddings/chemberta.csv",
        '3dinfomax': "/home/stallam_umass_edu/new_changes/polymertl/pre_trained_embeddings/3dinfomax.csv",
        'unimol': "/home/stallam_umass_edu/new_changes/polymertl/pre_trained_embeddings/unimol.csv",
        'morgan': "/home/stallam_umass_edu/new_changes/polymertl/pre_trained_embeddings/morgan_fingerprints.csv",
    }
    embedding_dict_2 = {
        'chemberta': "/home/stallam_umass_edu/new_changes/polymertl/pre_trained_embeddings/chemberta_2.csv",
        '3dinfomax': "/home/stallam_umass_edu/new_changes/polymertl/pre_trained_embeddings/3dinfomax_2.csv",
        'unimol': "/home/stallam_umass_edu/new_changes/polymertl/pre_trained_embeddings/unimol_2.csv",
        'morgan': "/home/stallam_umass_edu/new_changes/polymertl/pre_trained_embeddings/morgan_fingerprints_2.csv",
    }
    
    if embedding_choice in ['chemberta', '3dinfomax', 'unimol', 'morgan']:
        if dataset_type == 'copolymers':
            embedding_path = embedding_dict[embedding_choice]
        elif dataset_type == 'copolymers_2':
            embedding_path = embedding_dict_2[embedding_choice]
    else:
        print("Unknown embedding choice")
        return 
    
    nbits, embeddings = load_embeddings(embedding_path)
    df = load_dataset(data_path, dataset_type)
    
    if prop == 'ip':
        if dataset_type == 'copolymers':
            prop = "IP"
        elif dataset_type == 'copolymers_2':
            prop = "IP_vs_SHE"
    elif prop == 'ea':
        if dataset_type == 'copolymers':
            prop = "EA"
        elif dataset_type == 'copolymers_2':
            prop = "EA_vs_SHE"
    elif prop == "os":
        if dataset_type != 'copolymers_2':
            print("only copolymers_2 has oscillator_strength")
            return
        prop = "oscillator_strength"
    elif prop == 'og':
        if dataset_type != 'copolymers_2':
            print("only copolymers_2 has oscillator_strength")
            return
        prop = "optical_gap"
    else:
        print("unknown property, please use either 'ip', 'ea', 'os' or 'og'.")
        return
    
    if model_choice not in ['RNN', 'MLP']:
        print("unkown model")
        return
        
    print("using the Model: ", model_choice)
    print("using the property: ", prop, '\n')
    
    df = df[df['property'] == prop]
    ncols = len(list(df.columns))
    
    current_time = time.strftime("%Y%m%d-%H%M%S")
    model_path = f"models/model_{model_choice}_{prop}_{current_time}.pt"
    
    nbits, X, y = get_repeated_polymer(df, embeddings, nbits, repetition_format, dataset_type, repetitions)
    
    print(X.shape, '\n')
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)
    y_temp.reset_index(drop = True, inplace = True)
    
    X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype = torch.float32)
    test_dataset = MolecularDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    
    kfold = KFold(n_splits = 10, shuffle = True, random_state = 11)
    
    fold_no = 1
    r2s = []
    maes = []
    rmses = []
    for train, val in kfold.split(X_temp, y_temp):
        X_train, X_val = X_temp[train], X_temp[val]
        y_train, y_val = y_temp[train], y_temp[val]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

        train_dataset = MolecularDataset(X_train_tensor, y_train_tensor)
        val_dataset = MolecularDataset(X_val_tensor, y_val_tensor)        

        shuffle = True
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        
        if model_choice == 'RNN':
            seq_len = repetitions
            emb_dim = nbits
            out_dim = 1

            model = model = RNN(seq_len, emb_dim, lstm_dim, linear_dim, out_dim).to(device)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

            train_losses, test_losses = train_RNN(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, num_epochs, patience, device, wandb_log = False, model_path = model_path)

        # Evaluate the model
        predicted, targets = evaluate_model(model, test_dataloader, device)

        # Calculate R2 score, MAE, and RMSE
        r2 = r2_score(targets, predicted)
        mae = mean_absolute_error(targets, predicted)
        rmse = np.sqrt(mean_squared_error(targets, predicted))

        print(f"R2 Score: {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")

        # Check if the file exists, if not, create the file and write the header
        if not os.path.exists(file_name):
            with open(file_name, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(['Embedding', 'prop', 'fold_no', 'R2', 'MAE', 'RMSE'])

        # Append the evaluation metrics to the file
        with open(file_name, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([embedding_choice, prop, fold_no, r2, mae, rmse])
        fold_no += 1
        
        r2s.append(r2)
        maes.append(mae)
        rmses.append(rmse)

    r2s = np.array(r2s)
    maes = np.array(maes)
    rmses = np.array(rmses)
    
    # Check if the file exists, if not, create the file and write the header
    if not os.path.exists(file_name_review):
        with open(file_name_review, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['Embedding', 'prop', 'R2_mean', 'R2_std', 'R2_max', 'R2_min', 'MAE_mean', 'MAE_std', 'MAE_max', 'MAE_min', 'RMSE_mean', 'RMSE_std', 'RMSE_max', 'RMSE_min'])

    # Append the evaluation metrics to the file
    with open(file_name_review, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([embedding_choice, prop, np.mean(r2s), np.std(r2s), np.max(r2s), np.min(r2s), np.mean(maes), np.std(maes), np.max(maes), np.min(maes), np.mean(rmses), np.std(rmses), np.max(rmses), np.min(rmses)])
    
    del model, train_dataloader, test_dataloader, val_dataloader, predicted, targets
    del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
    del train_dataset, val_dataset, test_dataset, df, embeddings
    del X_train, X_test, y_train, y_test, X_val, y_val
    torch.cuda.empty_cache()
    gc.collect()
    
    return

if __name__ == '__main__':
    
    embedding_types = ['3dinfomax', 'unimol', 'morgan', 'chemberta']
    props = ["os", "og"]

    for prop in props:
        for embedding_choice in embedding_types:
            print("Property : ", prop)
            print("Embeddings picked : ", embedding_choice, '\n')
            train(embedding_choice, prop)