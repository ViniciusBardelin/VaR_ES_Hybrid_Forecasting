import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras_tuner.tuners import BayesianOptimization
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("ins_data.csv", nrows=2500)
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)

feature_col = "Sigma2_GARCH"
target_col  = "RV_APPLE"
lookback = 22

n = len(data)
val_frac = 0.2
val_start = int(n * (1 - val_frac))

train_df = data.iloc[:val_start].copy()
val_df   = data.iloc[val_start - lookback:].copy()  

x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))  

train_x = x_scaler.fit_transform(train_df[[feature_col]].values)
train_y = y_scaler.fit_transform(train_df[[target_col]].values)

val_x   = x_scaler.transform(val_df[[feature_col]].values)
val_y   = y_scaler.transform(val_df[[target_col]].values)

def make_windows(x_2d, y_2d, lookback: int):
    X, y = [], []
    for t in range(lookback, len(x_2d)):
        X.append(x_2d[t - lookback:t, :])
        y.append(y_2d[t, :])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = make_windows(train_x, train_y, lookback)
X_val,   y_val   = make_windows(val_x,   val_y,   lookback)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape,   "y_val:  ", y_val.shape)

def build(hp):
    activation = hp.Choice('activation', ['relu','tanh','linear','selu','elu'])
    recurrent_dropout = hp.Float('recurrent_dropout', 0.0, 0.5, default=0.2)
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, default=1)

    model = Sequential()

    for i in range(num_layers):
        units_i = hp.Int(f'units_lstm_{i+1}', min_value=16, max_value=256, step=16, default=64)
        return_seq = (i < num_layers - 1)

        if i == 0:
            model.add(LSTM(
                units=units_i,
                activation=activation,
                recurrent_dropout=recurrent_dropout,
                return_sequences=return_seq,
                input_shape=(X_train.shape[1], 1)
            ))
        else:
            model.add(LSTM(
                units=units_i,
                activation=activation,
                recurrent_dropout=recurrent_dropout,
                return_sequences=return_seq
            ))

    model.add(Dense(1))

    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mse']
    )
    return model

bayesian_opt_tuner = BayesianOptimization(
    build,
    objective='val_mse',
    max_trials=20,
    executions_per_trial=1,
    directory=r'C:\keras_tuning',
    project_name='kerastuner_bayes',
    overwrite=True
)

n_epochs= 100

bayesian_opt_tuner.search(X_train, y_train, epochs = n_epochs,
                          validation_split = 0.2,
                          verbose = 1)

best_hp = bayesian_opt_tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hp.values)

model.summary()

best = best_hp.values
print("activation:", best["activation"])
print("recurrent_dropout:", best["recurrent_dropout"])
print("learning_rate:", best["learning_rate"])
print("num_layers:", best["num_layers"])
for i in range(1, best["num_layers"] + 1):
    print(f"units_lstm_{i}:", best[f"units_lstm_{i}"])

#activation: selu
#recurrent_dropout: 0.1599489955842322
#learning_rate: 0.001
#num_layers: 1
#units_lstm_1: 176