import numpy as np
import pandas as pd


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


def get_repeated_polymer(df, embeddings, nbits, repetition_format=None, homopolymer=True, repetitions=400):
    if homopolymer:
        fp = df['mono'].apply(lambda smile: embeddings[smile])
    else:
        unique_archs = df['chain_arch'].nunique()
        if repetition_format == "weighted_add":
            df = pd.get_dummies(df, prefix=['chain_arch'], columns=['chain_arch'])
            nbits += unique_archs
            fp = df.apply(lambda x: np.append((np.array(embeddings[x['monoA']]) * x['fracA'] + np.array(embeddings[x['monoB']]) * x['fracB']), [x['chain_arch_alternating'], x['chain_arch_block'], x['chain_arch_random']]), axis=1)
        elif repetition_format == "concat":
            fp = df.apply(lambda x: concatenate_embeddings(x, embeddings, repetitions), axis=1)
    
    fp = list(fp)
    fp_array = np.asarray([np.array(fp[i]) for i in range(len(fp))])

    repeated_polymer = fp_array.reshape(len(df), repetitions, nbits)
    
    return nbits, repeated_polymer, df['value']