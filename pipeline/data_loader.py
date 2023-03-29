import os
import pandas as pd

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
    if dataset_name == 'homopolymers' or dataset_name == 'copolymers':
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