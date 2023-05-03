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
import wandb
from tqdm import tqdm
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import subprocess
import gc
import csv

gpu_index = 1
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

os.environ["WANDB_SILENT"] = "false"
os.environ["WANDB_IGNORE_ERRORS"] = "false"

def get_wandb_api_key():
    try:
        api_key = os.environ.get("WANDB_API_KEY")
    except Exception as e:
        print(e)
        api_key = None
    return api_key
WAND_API_KEY = get_wandb_api_key()

def try_wandb_login():
    WAND_API_KEY = get_wandb_api_key()
    if WAND_API_KEY:
        try:
            subprocess.run(["wandb", "login", WAND_API_KEY], check=True)
            return True
        except Exception as e:
            print(e)
            return False
    else:
        print("WARNING: No wandb API key found, this run will NOT be logged to wandb.")
        input("Press any key to continue...")
        return False
wandb_con = try_wandb_login()

# Save evaluation metrics to a unique CSV file
current_time = time.strftime("%Y%m%d-%H%M%S")
file_name = f"results/evaluation_metrics_{current_time}.csv"


def train():
    
    # Initialize a new wandb run
    run = wandb.init()
    
     # Get the hyperparameters from the Sweep
    config = wandb.config
    learning_rate = config.learning_rate
    num_epochs = config.num_epochs
    weight_decay = config.weight_decay
    model_choice = config.model_choice
    data_path = config.data_path
    embedding_path = config.embedding_path
    dataset_type = config.dataset
    repetitions = config.repetitions
    repetition_format = config.rep_style
    prop = config.prop
    batch_size = config.batch_size
    test_portion = config.test_portion
    lstm_dim = config.lstm_dim
    linear_dim = config.linear_dim
    scheduler_choice = config.scheduler
    hidden_dim = config.hidden_dim
    num_layers = config.num_layers
    
    
    
    # Set the run name with hyperparameter details
    run_name = f"{scheduler_choice}_{learning_rate}"
    wandb.run.name = run_name
    wandb.run.save()
    
    flag = False
    if dataset_type == "homopolymers":
        flag = True
    
    nbits, embeddings = load_embeddings(embedding_path)
    df = load_dataset(data_path, dataset_type)
    
    if prop == 'ip':
        if dataset_type == 'copolymers':
            prop = "IP" #"Ei"
        elif dataset_type == "homopolymers":
            prop = "Ei"
    elif prop == 'ea':
        if dataset_type == 'copolymers':
            prop = "EA" #"Eea"
        elif dataset_type == "homopolymers":
            prop = "Eea"
    else:
        print("unknown property, please use either 'ip' or 'ea'.")
        return
    
    if model_choice not in ['RNN', 'MLP']:
        print("unkown model")
        return
        
    print("using the Model: "+ model_choice)
    print("using the property: "+ prop)
    df = df[df['property'] == prop]
    ncols = len(list(df.columns))
    
    nbits, Mix_X_100Block, target = get_repeated_polymer(df, embeddings, nbits, repetition_format, flag, repetitions)
    
    X_train, X_test, y_train, y_test = train_test_split(Mix_X_100Block, target, test_size=test_portion, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=11)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    
    train_dataset = MolecularDataset(X_train_tensor, y_train_tensor)
    val_dataset = MolecularDataset(X_val_tensor, y_val_tensor)
    test_dataset = MolecularDataset(X_test_tensor, y_test_tensor)
    
    batch_size = batch_size
    shuffle = True
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    num_epochs = num_epochs
    if model_choice == 'RNN':
        seq_len = repetitions
        emb_dim = nbits
        lstm_dim = lstm_dim
        linear_dim = linear_dim

        out_dim = 1
        learning_rate = learning_rate
        weight_decay = weight_decay 
        
        model = model = RNN(seq_len, emb_dim, lstm_dim, linear_dim, out_dim).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        
        ### Create the learning rate scheduler
        if scheduler_choice == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_choice == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_choice == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        elif scheduler_choice == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        elif scheduler_choice == 'Constant':
            scheduler = None
        
        train_losses, test_losses = train_RNN(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, num_epochs, device, wandb_log = True)
        
    elif model_choice == 'MLP':
        input_dim = nbits
        hidden_dim = hidden_dim #100
        num_layers = num_layers

        output_dim = 1
        learning_rate = learning_rate

        model = MLP(input_dim, hidden_dim, output_dim, num_layers=num_layers).to(device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        ### Initialize the scheduler based on the scheduler_choice
        if scheduler_choice == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_choice == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_choice == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        elif scheduler_choice == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        elif scheduler_choice == 'Constant':
            scheduler = None
        
        train_losses, test_losses = train_MLP(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, num_epochs, device, wandb_log = True)
    
    current_time = time.strftime("%Y%m%d-%H%M%S")
    model_path = f"models/model_{model_choice}_{prop}_{current_time}.pt"
    
    ##### stop saving to save space
    torch.save(model.state_dict(), model_path)
    print("Saved PyTorch Model State to "+model_path)

    # model evaluation
    # Evaluate the model
    predicted, targets = evaluate_model(model, test_dataloader, device)

    # Calculate R2 score, MAE, and RMSE
    r2 = r2_score(targets, predicted)
    mae = mean_absolute_error(targets, predicted)
    rmse = np.sqrt(mean_squared_error(targets, predicted))

    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    
    wandb.log({"R2 Score": r2, "Mean Absolute Error": mae, "Root Mean Squared Error": rmse})
        
    # Check if the file exists, if not, create the file and write the header
    if not os.path.exists(file_name):
        with open(file_name, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['Scheduler', 'Learning Rate', 'R2', 'MAE', 'RMSE'])

    # Append the evaluation metrics to the file
    with open(file_name, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([scheduler_choice, learning_rate, r2, mae, rmse])
        
    del model, train_dataloader, test_dataloader, val_dataloader, predicted, targets
    del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
    del train_dataset, val_dataset, test_dataset, df, embeddings
    del X_train, X_test, y_train, y_test, X_val, y_val,  Mix_X_100Block, target
    torch.cuda.empty_cache()
    gc.collect()
    
    return

    

if __name__ == '__main__':
    sweep_config = {
        "name": "Sweep",
        "method": "grid",
        "metric": {
            "name": "test_loss",
            "goal": "minimize",
        },
        "parameters": {
            "prop": {
                "value": 'ip', #"ea"
            },
            "model_choice": {
                "value": "MLP",# MLP, RNN 
                ##change rep_style to weighted_add if MLP.
            },
            "learning_rate": {
                #"value": 0.001,
                "values": [0.01, 0.001, 0.0001, 1e-5, 1e-6]
            },
            "weight_decay": {
                "value": 1e-6,
                #"values" : [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
            },
            "batch_size": {
                "value": 128,
            },
            "num_epochs": {
                "value": 100,
            },
            "data_path": {
                "value": "/home/zmao_umass_edu/final_dataset/copolymers",
            },
            "embedding_path": {
                "value": "/home/zmao_umass_edu/pre_trained_embeddings/chemberta.csv",
            },
            "dataset": {
                "value": "copolymers",
            },
            "repetitions": {
                "value": 1, #100
            },
            "rep_style": {
                #MLP uses weighted_add only.
                "value": "weighted_add", #"weighted_add, concat"
            },
            "test_portion": {
                "value": 0.2,
                #"values": [0.5, 0.6, 0.7, 0.8, 0.9]
            },
            "lstm_dim": {
                "value": 20,
                #"values": [512]
            },
            "linear_dim": {
                "value": 32,
                #"values": [128]
            },
            "scheduler": {
                #"value": "CosineAnnealingLR",
                "values": ["StepLR", "ExponentialLR", "CosineAnnealingLR", "ReduceLROnPlateau", "Constant"]
            },
            "hidden_dim": {
                "value": 256,
                #"values": [32, 64, 100, 128, 256, 512] 
                #"values": [128, 256, 512, 768, 1024] 
                #"values": [32, 64, 100]
            },
            "num_layers": {
                "value": 10,
                #"values": [2, 3, 4, 5, 6, 7, 8, 9, 10]
                #"values": [6, 7, 8, 9, 10]
            },
        },
    }
    
    sweep_id = wandb.sweep(sweep_config, project="MLP_Scheduler_lr-test")
    wandb.agent(sweep_id, train) #count = 5