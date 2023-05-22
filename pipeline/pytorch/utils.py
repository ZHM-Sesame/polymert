import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_dataset(dataset_path, dataset_name):
    try:
        if dataset_path.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(dataset_path)
        else:
            df = pd.read_csv(dataset_path + '/' + dataset_name + '.csv') 
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # 'copolymers' schema: monoA,monoB,fracA,fracB,chain_arch,property,value
    # 'homopolymers' schema: mono,property,value
    if dataset_name in ['homopolymers', 'copolymers', 'copolymers_2']:
        return df
    elif dataset_name == 'both':
        homopolymer_df = load_dataset(dataset_path, 'homopolymers')
        copolymer_df = load_dataset(dataset_path, 'copolymers')
        return homopolymer_df, copolymer_df
    
    print(f"We only support 3 types of datasets: 'homopolymers', 'copolymers' and 'both', please rename your files accordingly")
    return None
        

def load_embeddings(embedding_path):
    """
    Reads the embeddings based on their path and returns a dictionary (smiles => embeddings).
    Supports CSV files for now.
    """
    if os.path.isdir(embedding_path):
        raise ValueError("Please provide the complete path to the embedding file.")
    else:
        try:
            # Read CSV file
            df = pd.read_csv(embedding_path)
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    cols = list(df.columns)[1:]
    nbits = len(cols)
    df['embeddings'] = df[cols].values.tolist()
    emb_dict = df.set_index('smiles')['embeddings'].to_dict()
    
    return nbits, emb_dict

def concatenate_embeddings(row, embeddings, repetitions):
    """
    Arranges two input lists a and b into a list of lists of length l,
    according to the given ratio and arrangement type.
    Ratio is a float value between 0 and 1 representing the proportion of a to b
    in the final list of lists.
    Arrangement type can be 'random', 'block', or 'alternating'.
    """
    mul_a = 1 if row['fracA'] < row['fracB'] else int(row['fracA'] / row['fracB'])
    mul_b = 1 if row['fracB'] < row['fracA'] else int(row['fracB'] / row['fracA'])

    arrangement_type = row['chain_arch']
    
    emb_a, emb_b = np.array(embeddings[row['monoA']]), np.array(embeddings[row['monoB']])
        
    rep_a = np.tile(emb_a, (mul_a, 1))
    rep_b = np.tile(emb_b, (mul_b, 1))
    rep_unit = np.concatenate((rep_a, rep_b), axis = 0)
    repeated_polymer = np.tile(rep_unit , (int(repetitions / (mul_a + mul_b)), 1))

    if repetitions % (mul_a + mul_b) != 0:
        repeated_polymer = np.concatenate((repeated_polymer, np.tile(emb_a, (repetitions % (mul_a + mul_b), 1))), axis = 0)
    
    if arrangement_type == 'random':
        np.random.shuffle(repeated_polymer)
    
    return repeated_polymer


def get_repeated_polymer(df, embeddings, nbits, repetition_format=None, dataset_type="copolymers", repetitions=400):
    if dataset_type == "homopolymers":
        fp = df['mono'].apply(lambda smile: embeddings[smile])
    else:
        unique_archs = df['chain_arch'].nunique()
        if repetition_format == "weighted_add":
            df = pd.get_dummies(df, prefix=['chain_arch'], columns=['chain_arch'])
            nbits += unique_archs
            if dataset_type == "copolymers":
                fp = df.apply(lambda x: np.append((np.array(embeddings[x['monoA']]) * x['fracA'] + np.array(embeddings[x['monoB']]) * x['fracB']), [x['chain_arch_alternating'], x['chain_arch_block'], x['chain_arch_random']]), axis=1)
            elif dataset_type == "copolymers_2":
                fp = df.apply(lambda x: np.append((np.array(embeddings[x['monoA']]) * x['fracA'] + np.array(embeddings[x['monoB']]) * x['fracB']), [x['chain_arch_alternating']]), axis=1)
            
        elif repetition_format == "concat":
            fp = df.apply(lambda x: concatenate_embeddings(x, embeddings, repetitions), axis=1)
    
    fp = list(fp)
    fp_array = np.asarray([np.array(fp[i]) for i in range(len(fp))])

    repeated_polymer = fp_array.reshape(len(df), repetitions, nbits)
    
    return nbits, repeated_polymer, df['value']


class MolecularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        x = x.clone().detach().to(dtype=torch.float32)
        y = y.clone().detach().to(dtype=torch.float32)
        
        return x, y

def evaluate_model(model, dataloader, device):
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            # x = x.to(device)
            # y = y.to(device)
            x = x.to(device).float()
            y = y.to(device).float()
            y = y.view(-1, 1)
            
            output = model(x)
            #output = output.squeeze(dim=2)

            all_outputs.extend(output.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    return np.array(all_outputs), np.array(all_targets)





