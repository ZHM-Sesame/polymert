from data_loader import load_dataset, load_embeddings
from data_transformer import get_repeated_polymer

import time, random, argparse, os, pandas as pd, datetime, numpy as np, pickle

from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional, TimeDistributed, Reshape, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

def get_hp_settings():
    wandb.init(
        project="polymertl",
        config={
            "loss": 'mse',
            "optimizer": 'adam',
            "epoch": 100,
            "batch_size": 32
        }
    )
    return wandb.config


def trainRNN(args):
    
    config = get_hp_settings()
    
    data_path = args.data
    embedding_path = args.embedding
    dataset_type = args.dataset
    repetitions = args.repetitions
    repetition_format = args.rep_style
    
    flag = False
    if dataset_type == "homopolymers":
        flag = True
    
    nbits, embeddings = load_embeddings(embedding_path)
    DF = load_dataset(data_path, dataset_type)
    
    prop = ''
    if args.prop == 'ip':
        prop = "IP" #"Ei"
    elif args.prop == 'ea':
        prop = "EA" #"Eea"
    else:
        print("unknown property, please use either 'ip' or 'ea'.")
        return
        
    print("using the property: "+ prop)
    DF = DF[DF['property'] == prop]
    ncols = len(list(DF.columns))
    
    nbits, Mix_X_100Block, target = get_repeated_polymer(DF, embeddings, nbits, repetition_format, flag, repetitions)
    X_train, X_test, y_train, y_test = train_test_split(Mix_X_100Block, target, test_size=0.2, shuffle=False)
    X_train = X_train.astype('float')
    y_train = y_train.astype('float')
    
    LSTMunits = 30
    RNNmodel = Sequential()
    
    RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True), input_shape=(repetitions, nbits)))
    RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
    RNNmodel.add(TimeDistributed(Dense(int(LSTMunits/2), activation="relu")))
    RNNmodel.add(Reshape((int(LSTMunits/2*repetitions),)))
    RNNmodel.add(Dense(1))
    RNNmodel.compile(loss=config.loss, optimizer=config.optimizer)
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
    RNNmodel.fit(X_train, y_train, validation_split=0.2, epochs=config.epoch, batch_size=config.batch_size, 
                 callbacks=[es, WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")])
    
    wandb.finish()
    
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists('models'):
        os.makedirs('models')
    filepath = 'models/RNN_'+args.prop+'_'+time+'.model'
    save_model(RNNmodel, filepath, save_format='h5')

    # model evaluation
    print("model performance " + prop)
    y_pred_train = RNNmodel.predict(X_train, batch_size = 32)
    print("Train set R^2: %.2f" % r2_score(y_train, y_pred_train))
    print("Train MAE score: %.2f" % mean_absolute_error(y_train, y_pred_train))
    print("Train RMSE score: %.2f" % np.sqrt(mean_squared_error(y_train, y_pred_train)))

    y_pred_test = RNNmodel.predict(X_test, batch_size = 32)
    print("Test set R^2: %.2f" % r2_score(y_test, y_pred_test))
    print("Test MAE score: %.2f" % mean_absolute_error(y_test, y_pred_test))
    print("Test RMSE score: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))
    
    return

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default='RNN', help='Choose either "MUL" or"RNN"')
    parser.add_argument('--prop', type=str, required=False, default='ip', help='ip or ea')
    parser.add_argument('--data', type=str, required=False, default='../final_dataset/copolymers', help='data directory')
    parser.add_argument('--dataset', type=str, required=False, default='copolymers', help='=> "copolymers" or "homopolymers"')
    parser.add_argument('--embedding', type=str, required = False, default = '../pre_trained_embeddings/3dinfomax.csv', help='embedding type')
    #parser.add_argument('--embedding', type=str, required=False, default='../final_dataset/polymer_fingerprint_mappings/polymer_fingerprint_mappings.csv', help='embedding type')
    parser.add_argument('--repetitions', type=int, required=False, default=50, help='repetitions of the monomer unit')
    parser.add_argument('--rep_style', type=str, required=False, default='concat', help='weighted_add or concat')
    parsed_args = parser.parse_args()

    trainRNN(parsed_args)