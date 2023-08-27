from utils import load_dataset, load_embeddings, get_repeated_polymer, MolecularDataset, evaluate_model
from rnn import MolecularRNNDataset, RNN, train_RNN

import pandas as pd, numpy as np, pickle, time, os, torch, gc, csv
import matplotlib.pyplot as plt
import torch.nn.init as init

from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

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
    dataset_type = "copolymers"
    batch_size = 64
    patience = 10
    
    if dataset_type in ["copolymers", "copolymers_2"]:
        lstm_dim = 128
        linear_dim = 32
        learning_rate = 0.001
        scheduler_choice = 'CosineAnnealingLR'
        repetitions = 100
        repetition_format = "concat"
    
    embedding_dict = {
        'chemberta': "/home/stallam_umass_edu/new_changes/polymertl/pre_trained_embeddings/chemberta.csv",
        '3dinfomax': "/home/stallam_umass_edu/new_changes/polymertl/pre_trained_embeddings/3dinfomax.csv",
        'unimol': "/home/stallam_umass_edu/new_changes/polymertl/pre_trained_embeddings/unimol.csv",
        'morgan': "/home/stallam_umass_edu/new_changes/polymertl/pre_trained_embeddings/morgan_fingerprints.csv",
    }
    
    embedding_path = embedding_dict[embedding_choice]
    nbits, embeddings = load_embeddings(embedding_path)
    df = load_dataset(data_path, dataset_type)
    
    if prop == 'ip':
        prop = "IP"
    elif prop == 'ea':
        prop = "EA"
    else:
        print("unknown property, please use either 'ip', 'ea', 'os' or 'og'.")
        return
        
    print("using the Model: ", model_choice)
    print("using the property: ", prop, '\n')
    
    df = df[df['property'] == prop]
    ncols = len(list(df.columns))
    
    current_time = time.strftime("%Y%m%d-%H%M%S")
    model_path = f"models/model_{model_choice}_{prop}_{current_time}.pt"
    
    nbits, X, y = get_repeated_polymer(df, embeddings, nbits, repetition_format, dataset_type, repetitions)
    print(X.shape, '\n')
    
    cv = KFold(n_splits = 10, shuffle = True, random_state = 42)
    folds = []
    for train_idx, test_idx in cv.split(X, y):
        folds.append([train_idx, test_idx])
        
    r2s, maes, rmses = [], [], []
    for i, fold in enumerate(folds):
        train, test = fold
        val = folds[i-1][1]
        train = [x for x in train if x not in val]
        
        X_train, X_val, X_test = X[train], X[val], X[test]
        y_train, y_val, y_test = y[train], y[val], y[test]
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype = torch.float32)

        train_dataset = MolecularDataset(X_train_tensor, y_train_tensor)
        val_dataset = MolecularDataset(X_val_tensor, y_val_tensor)
        test_dataset = MolecularDataset(X_test_tensor, y_test_tensor)
        
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
        
        seq_len = repetitions
        emb_dim = nbits
        out_dim = 1

        model = model = RNN(seq_len, emb_dim, lstm_dim, linear_dim, out_dim).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs, eta_min = 0)

        train_losses, val_losses = train_RNN(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, num_epochs, patience, device, wandb_log = False, model_path = model_path)
        predicted, targets = evaluate_model(model, test_dataloader, device)

        r2 = r2_score(targets, predicted)
        mae = mean_absolute_error(targets, predicted)
        rmse = np.sqrt(mean_squared_error(targets, predicted))

        if not os.path.exists(file_name):
            with open(file_name, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(['Embedding', 'prop', 'fold_no', 'R2', 'MAE', 'RMSE'])

        with open(file_name, mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([embedding_choice, prop, i + 1, r2, mae, rmse])
        
        r2s.append(r2)
        maes.append(mae)
        rmses.append(rmse)
        
        del model, train_dataloader, test_dataloader, val_dataloader, predicted, targets
        del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
        del train_dataset, val_dataset, test_dataset
        del X_train, X_test, y_train, y_test, X_val, y_val
        

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
    
    del df, embeddings
    torch.cuda.empty_cache()
    gc.collect()
    
    return

if __name__ == '__main__':
    
    # embedding_types = ['3dinfomax', 'unimol', 'morgan', 'chemberta']
    # props = ["ip", "ea"]
    
    embedding_types = ['morgan', 'chemberta']
    props = ["ea"]

    for prop in props:
        for embedding_choice in embedding_types:
            print("Property   : ", prop)
            print("Embeddings : ", embedding_choice, '\n')
            train(embedding_choice, prop)