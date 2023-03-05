import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import keras_tuner as kt
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Dense, Flatten, Activation, ZeroPadding2D
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional, TimeDistributed, Reshape, Dropout
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
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

def train(args):

    #read in the reference data
    DF = pd.read_csv('datasets/Dataset 1.csv') #to be comparable with the size of other datasets, select ~5000 data points from the reference
    DF['TRIMER_mol'] = DF['TRIMER'].apply(Chem.MolFromSmiles)
    DF = DF.dropna()

    
    #fingperints featurization
    fp = DF['TRIMER_mol'].apply(lambda m: AllChem.GetMorganFingerprint(m, radius=3))
    #Recognize all n substructures found in the datasets
    fp_n = fp.apply(lambda m: m.GetNonzeroElements())
            
    HashCode = []
    for i in fp_n:
        for j in i.keys():
            HashCode.append(j)
    unique_set = set(HashCode)
    unique_list = list(unique_set)
    Corr_df = pd.DataFrame(unique_list).reset_index()
    
    if not os.path.exists('MY_finger_dataset.csv'):
        MY_finger = []
        for polymer in fp_n:
            my_finger = [0] * len(unique_list)
            for key in polymer.keys():
                index = Corr_df[Corr_df[0] == key]['index'].values[0]
                my_finger[index] = polymer[key]
            MY_finger.append(my_finger)
        MY_finger_dataset = pd.DataFrame(MY_finger)
        MY_finger_dataset.to_csv('MY_finger_dataset.csv', index=False)
    else:
        MY_finger_dataset=pd.read_csv('MY_finger_dataset.csv').reset_index()

    # filter feature matrix using only the most dominant substructures in the dataset
    Zero_Sum = (MY_finger_dataset == 0).astype(int).sum()
    NumberOfZero = 4670 # a optimized parameter for feature filter: if more than 4670 of the 5000 samples do not pocess one substrcture, then remove this substructure from the feature matrix
    X = MY_finger_dataset[Zero_Sum[Zero_Sum < NumberOfZero].index]
    
    # RNN model

    # data split into train/test sets for property IP
    X_train, X_test, y_train, y_test = train_test_split(fp, DF['IP (eV)'], test_size=0.2, random_state=11)

    # model setup using the optimized architecture for IP
    LSTMunits = 20 # hyperprameter for LSTM
    RNNmodel = Sequential()
    RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True), input_shape=(100,132)))
    RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
    RNNmodel.add(TimeDistributed(tensorflow.keras.layers.Dense(int(LSTMunits/2), activation="relu")))
    RNNmodel.add(Reshape((int(LSTMunits/2*100),)))
    RNNmodel.add(Dense(1))

    RNNmodel.compile(loss='mse', optimizer='adam')
    RNNmodel.fit(X_train, y_train, validation_split=0.2, epochs=120, batch_size=64)

    filepath = 'Binary_IP_RNN.model'
    save_model(model, filepath, save_format='h5')

    # model evaluation
    print("model performance (IP)")
    y_pred_train = RNNmodel.predict((X_train))
    print("Train set R^2: %.2f" % r2_score(y_train, y_pred_train))
    print("Train MAE score: %.2f" % mean_absolute_error(y_train, y_pred_train))
    print("Train RMSE score: %.2f" % np.sqrt(mean_squared_error(y_train, y_pred_train)))
    y_pred_test = RNNmodel.predict((X_test))
    print("Test set R^2: %.2f" % r2_score(y_test, y_pred_test))
    print("Test MAE score: %.2f" % mean_absolute_error(y_test, y_pred_test))
    print("Test RMSE score: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))


    # data split into train/test sets for property EA
    X_train, X_test, y_train, y_test = train_test_split(Mix_X_100Block, DF['EA (eV)'], test_size=0.2, random_state=11)

    # model setup using the optimized architecture for EA
    LSTMunits = 20 # hyperprameter for LSTM
    RNNmodel = Sequential()
    RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True), input_shape=(100,132)))
    RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
    RNNmodel.add(TimeDistributed(Dense(int(LSTMunits/2), activation="relu")))
    RNNmodel.add(Reshape((int(LSTMunits/2*100),)))
    RNNmodel.add(Dense(1))

    RNNmodel.compile(loss='mse', optimizer='adam')
    RNNmodel.fit(X_train, y_train, validation_split=0.2, epochs=120, batch_size=64)

    filepath = 'Binary_EA_RNN.model'
    save_model(model, filepath, save_format='h5')

    # model evaluation
    print("model performance (EA)")
    y_pred_train = RNNmodel.predict((X_train))
    print("Train set R^2: %.2f" % r2_score(y_train, y_pred_train))
    print("Train MAE score: %.2f" % mean_absolute_error(y_train, y_pred_train))
    print("Train RMSE score: %.2f" % np.sqrt(mean_squared_error(y_train, y_pred_train)))
    y_pred_test = RNNmodel.predict((X_test))
    print("Test set R^2: %.2f" % r2_score(y_test, y_pred_test))
    print("Test MAE score: %.2f" % mean_absolute_error(y_test, y_pred_test))
    print("Test RMSE score: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required = True, 
    	help='Choose either "CNN", "DNN", "RNN", or "Fusion" for model architecture')
   
    parsed_args = parser.parse_args()

    train(parsed_args)