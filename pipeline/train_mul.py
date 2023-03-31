import os
import ast
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow as tf
from datetime import datetime 
from tensorflow.python.keras.engine import data_adapter
from keras_tuner import HyperModel
from keras_tuner.tuners import Hyperband
import tensorflow_addons as tfa
import IPython
from sklearn.metrics import mean_squared_error, r2_score

tf.random.set_seed(123)

def get_dataset(df):
    # return TF data set
    fps_mix = np.stack(df.fps_mix.values).astype(np.float32)
    #selector = np.stack(df.dummy.values).astype(np.float32)   
    selector = np.stack([np.fromstring(dummy, sep=' ', dtype=np.float32) for dummy in df.dummy.values])
    target = df.scaled_value.astype(np.float32)[:, np.newaxis]
    dataset = tf.data.Dataset.from_tensor_slices(({'sel': selector, 'fps_mix': fps_mix, 'prop': df.prop}, target))

    dataset = dataset.cache().batch(200).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


training_df, test_df = train_test_split(df, test_size=0.2, stratify=df.prop, random_state=123)
training_df, test_df = training_df.copy(), test_df.copy()

for train_index, val_index in skf.split(training_df, training_df.prop):
    # iterte over 5 splits
    train_df = training_df.iloc[train_index].copy()
    val_df = training_df.iloc[val_index].copy()
    
    # scale target values
    property_scaler = {}
    for prop in props:
        property_scaler[prop] = MinMaxScaler()

        # train
        cond = train_df[train_df.prop == prop].index
        train_df.loc[cond, ['scaled_value']] = property_scaler[prop].fit_transform(train_df.loc[cond, ['scaled_value']])
        
        # val
        cond = val_df[val_df.prop == prop].index
        val_df.loc[cond, ['scaled_value']] = property_scaler[prop].transform(val_df.loc[cond, ['scaled_value']])

    datasets.append({'train': get_dataset(train_df), 'val': get_dataset(val_df), 'property_scaler': property_scaler})

# Create final dataset for meta learner
property_scaler_final = {}
for prop in props:
    property_scaler_final[prop] = MinMaxScaler()
    
   # train
    cond = training_df[training_df.prop == prop].index
    training_df.loc[cond, ['scaled_value']] = property_scaler_final[prop].fit_transform(training_df.loc[cond, ['scaled_value']])

    # val
    cond = test_df[test_df.prop == prop].index
    test_df.loc[cond, ['scaled_value']] = property_scaler_final[prop].transform(test_df.loc[cond, ['scaled_value']])
    
datasets_final = {'train': get_dataset(training_df), 'test': get_dataset(test_df), 'property_scaler': property_scaler_final}


class PropertyDonwsteam(tfk.Model):
    def __init__(self, hp):
        super().__init__()
        # hp defines the hyper parameter
        self.my_layers = []
        self.concat_at = hp.Int('concat_at', 0, 2)
        
        for i in range(hp.Int('num_layers', 3, 3)): 
            new_step = [               
            tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=352,
                                            max_value=544,
                                            step=64),),
            
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dropout(hp.Float(
                'dropout_' + str(i),
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.1,
            )),
            ]

            self.my_layers.append(new_step)
        self.my_layers.append([tf.keras.layers.Dense(1)])

    def call(self, inputs):
        x = inputs['fps_mix']
        for num, layer_step in enumerate(self.my_layers):
            if self.concat_at == num:
                # concatenate the selector vector
                x = tf.concat((x, inputs['sel']), -1)
            for layer in layer_step:
                x = layer(x)
        return x
    
    def predict_step(self, data):
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)

        # drop prop here
        prop = x['prop']
        del x['prop']
        return self(x, training=False), data[-1], prop

    @tf.function
    def call_external(self, inputs):
        return self.call(inputs)


def build_model(hp):
    # returns the compiled tensorflow model
    model = PropertyDonwsteam(hp)
    opt = tf.keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-3]))
    opt = tfa.optimizers.SWA(opt)

    model.compile(
        optimizer=opt,
        loss='mse',)
    return model

def trainMul(args):
	# Read the data set
	df = pd.read_csv('../data/homopolymer_fps.csv')

	df = df.sample(frac=1, random_state=123)

	df['scaled_value'] = df['val']
	skf = StratifiedKFold(n_splits=5)

	props = df.prop.unique().tolist()

	datasets = []

	for num, data in enumerate(datasets):
		# iterate over all 5 VC data sets    
		tuner = Hyperband(
			build_model,
			objective='val_loss',
			max_epochs=10,#300
			seed=10,
			directory=f'hyperparameter_search_fp',
			project_name='fold_' + str(num)
			)

		reduce_lr = tfk.callbacks.ReduceLROnPlateau(
			factor=0.8,
			monitor="val_loss",
			verbose=1,
		)
		
		earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=40)
		
		class ClearTrainingOutput(tf.keras.callbacks.Callback):
			def on_train_end(*args, **kwargs):
				IPython.display.clear_output(wait = True)
		
		# Create an instance of the model and search the hyperspace
		tuner.search(data['train'],
					epochs=10,#300
					validation_data=data['val'],
					callbacks=[earlystop, reduce_lr, ClearTrainingOutput()],
					verbose=0
					)
		
		## Post processing: compute RMSE and R2 values (scale back before comnputation)
		best_values.append(tuner.get_best_hyperparameters()[0].values)
		best_model = tuner.get_best_models(1)[0]

		# Predict on the validataion data set
		res = np.concatenate(best_model.predict(data['val']), -1)
		
		# Save best model for later use in the meta learner
		best_model.save(f'models/fp/{num}', include_optimizer=False)

		# Compute RMSE and R2
		_df = pd.DataFrame(res, columns=['pred', 'target', 'prop'])
		_df['prop'] = _df.prop.apply(lambda x: x.decode('utf-8'))
		props = _df.prop.unique()
		
		property_scaler = data['property_scaler']
		for prop in props:

			cond = _df[_df.prop == prop].index
			rmse_scaled = mean_squared_error(_df.loc[cond, ['target']], _df.loc[cond, ['pred']], squared=False)
			r2_scaled = r2_score(_df.loc[cond, ['target']], _df.loc[cond, ['pred']])
			
			_df.loc[cond, ['pred']] = property_scaler[prop].inverse_transform(_df.loc[cond, ['pred']])
			_df.loc[cond, ['target']] = property_scaler[prop].inverse_transform(_df.loc[cond, ['target']])
			
			rmse = mean_squared_error(_df.loc[cond, ['target']], _df.loc[cond, ['pred']], squared=False)
			r2 = r2_score(_df.loc[cond, ['target']], _df.loc[cond, ['pred']])
			property_metric.append({'name': f'fp', 'prop': prop, 'rmse': rmse, 'r2':r2, 'fold': num, 'rmse_scaled': rmse_scaled, 'r2_scaled': r2_scaled})
			
		# Not scaled back
		rmse = mean_squared_error(res[:,0], res[:,1], squared=False)
		r2 = r2_score(res[:,0], res[:,1])
		
		results.append({'name': f'fp','r2': r2, 'rmse':rmse})

	df = pd.DataFrame(results)
	df
