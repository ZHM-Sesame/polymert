from data_loader import load_dataset, load_embeddings

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# import rdkit
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import Descriptors
# from rdkit.Chem import rdMolDescriptors
# from rdkit.Chem.Draw import IPythonConsole
# from rdkit.Chem import Draw
# from rdkit.Chem.Draw import rdMolDraw2D
# import keras_tuner as kt
# from keras_tuner.tuners import RandomSearch
# from keras_tuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Dense, Flatten, Activation, ZeroPadding2D
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional, TimeDistributed, Reshape, Dropout
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
from tensorflow.keras import layers
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import random
from numpy.random import seed
import tensorflow
from keras.layers import Input
from keras.models import Model
from keras.layers import Concatenate
import argparse
import os
import matplotlib.pyplot as plt
import datetime
import pandas as pd

def trainRNN(args):
    #read in the reference data
    #DF = pd.read_csv('datasets/Dataset 1.csv') #to be comparable with the size of other datasets, select ~5000 data points
    
    data_path = args.data
    embedding_path = args.embedding
    dataset_type = args.dataset
    
    # get embeddings
    nbits, embeddings = load_embeddings(embedding_path)
    
    # TODO: create different function for copolymers 
    # TODO: create arg for dataset and embedding types
    # mono, ea, ip
    DF = load_dataset(data_path, dataset_type)
    
    # data split into train/test sets
    prop = '' #select property to predict on.
    if args.prop == 'ip':
        #prop = "Ei"
        prop = "IP"
    elif args.prop == 'ea':
        #prop = "Eea"
        prop = "EA"
    else:
        print("unknown property, please use either 'ip' or 'ea'.")
        return
        
    print("using the property: "+ prop)
    DF = DF[DF['property'] == prop]
    ncols = len(list(DF.columns))
    
    # fingerprints featurization
    if(dataset_type == 'homopolymers'):
        fp = DF['mono'].apply(lambda m: embeddings[m])
        fp = list(fp)
    else:
        DF = pd.get_dummies(DF, prefix=['chain_arch'], columns=['chain_arch'])
        nbits += (len(list(DF.columns)) - ncols + 1)
        fp = DF.apply(lambda x: np.append((np.array(embeddings[x['monoA']]) * x['fracA'] + np.array(embeddings[x['monoB']]) * x['fracB']), [x['chain_arch_alternating'], x['chain_arch_block'], x['chain_arch_random']]), axis=1)
        fp = list(fp)
    
    fp_array = np.asarray([np.array(fp[i]) for i in range(len(fp))])
    
    repetitions = args.repetitions
    Mix_X_100Block = np.repeat(fp_array, repetitions, axis=0)
    Mix_X_100Block = Mix_X_100Block.reshape(len(DF), repetitions, nbits)
    X_train, X_test, y_train, y_test = train_test_split(Mix_X_100Block, DF['value'], test_size=0.2, random_state=11)
    
    X_train = X_train.astype('float')
    y_train = y_train.astype('float')
    
    # model setup using the optimized architecture for IP
    LSTMunits = 10 # hyperprameter for LSTM
    RNNmodel = Sequential()
    
    RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True), input_shape=(repetitions, nbits)))
    RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
    RNNmodel.add(TimeDistributed(tensorflow.keras.layers.Dense(int(LSTMunits/2), activation="relu")))
    RNNmodel.add(Reshape((int(LSTMunits/2*repetitions),)))
    RNNmodel.add(Dense(1))
    RNNmodel.compile(loss='mse', optimizer='adam')

    RNNmodel.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=32)
    
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists('models'):
        os.makedirs('models')
    filepath = 'models/RNN_'+args.prop+'_'+time+'.model'
    save_model(RNNmodel, filepath, save_format='h5')

    # model evaluation
    print("model performance " + prop)
    y_pred_train = RNNmodel.predict((X_train))
    print("Train set R^2: %.2f" % r2_score(y_train, y_pred_train))
    print("Train MAE score: %.2f" % mean_absolute_error(y_train, y_pred_train))
    print("Train RMSE score: %.2f" % np.sqrt(mean_squared_error(y_train, y_pred_train)))

    y_pred_test = RNNmodel.predict((X_test))
    print("Test set R^2: %.2f" % r2_score(y_test, y_pred_test))
    print("Test MAE score: %.2f" % mean_absolute_error(y_test, y_pred_test))
    print("Test RMSE score: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))
    
    return

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required = False, default = 'RNN', help='Choose either "MUL" or"RNN"')
    parser.add_argument('--prop', type=str, required = False, default = 'ip', help='ip or ea')
    parser.add_argument('--data', type=str, required = False, default = '../final_dataset/copolymers', help='data directory')
    parser.add_argument('--dataset', type=str, required = False, default = 'copolymers', help='=> "copolymers" or "homopolymers"')
    #parser.add_argument('--embedding', type=str, required = False, default = '../pre_trained_embeddings/chemberta.csv', help='embedding type')
    parser.add_argument('--embedding', type=str, required = False, default = '../final_dataset/polymer_fingerprint_mappings/polymer_fingerprint_mappings.csv', help='embedding type')
    parser.add_argument('--repetitions', type=int, required = False, default = 100, help='repetitions of the monomer unit')

    parsed_args = parser.parse_args()

    trainRNN(parsed_args)