import numpy as np
import pandas as pd

# repetition_format can be "weighted_add", "concat" 
def concatenate_embeddings(row, embeddings):
  mul_a = 1 if row['fracA'] < row['fracB'] else int(row['fracA'] / row['fracB'])
  mul_b = 1 if row['fracB'] < row['fracA'] else int(row['fracB'] / row['fracA'])
  emb_a = np.array(embeddings[row['monoA']])
  emb_b = np.array(embeddings[row['monoB']])
  rep_a = np.tile(emb_a, mul_a)
  rep_b = np.tile(emb_b, mul_b)

  if mul_a == mul_b:
    # alternating
    op = np.tile(np.concatenate((rep_a, rep_b), axis=None), 2)
  else:
    # block or random
    op = np.concatenate((rep_a, rep_b), axis=None)

  return op

def get_repeated_polymer(df, embeddings, nbits, repetition_format=None, homopolymer=True, repetitions=100):
  if homopolymer:
    fp = df['mono'].apply(lambda smile: embeddings[smile])
  else:
    unique_archs = df['chain_arch'].nunique()
    if repetition_format == "weighted_add":
      df = pd.get_dummies(df, prefix=['chain_arch'], columns=['chain_arch'])
      nbits += unique_archs
      fp = df.apply(lambda x: np.append((np.array(embeddings[x['monoA']]) * x['fracA'] + np.array(embeddings[x['monoB']]) * x['fracB']), [x['chain_arch_alternating'], x['chain_arch_block'], x['chain_arch_random']]), axis=1)
    elif repetition_format == "concat":
      df = df[df['chain_arch'] != 'random']
      # df = pd.get_dummies(df, prefix=['chain_arch'], columns=['chain_arch'])
      nbits = nbits * 4 # + unique_archs
      fp = df.apply(lambda x: concatenate_embeddings(x, embeddings), axis=1) #, [x['chain_arch_alternating'], x['chain_arch_block'], x['chain_arch_random']]), axis = 1)

  fp = list(fp)
  fp_array = np.asarray([np.array(fp[i]) for i in range(len(fp))])
  repeated_polymer = np.repeat(fp_array, repetitions, axis=0)
  repeated_polymer = repeated_polymer.reshape(len(df), repetitions, nbits)
  return nbits, repeated_polymer