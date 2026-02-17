import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras_tuner.tuners import BayesianOptimization
from keras_tuner.tuners import RandomSearch
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

# BAYESIAN OPTIMIZATION
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

# RANDOM SEARCH
def build(hp):
    activation = hp.Choice('activation', ['relu','tanh','linear','selu','elu'])
    recurrent_dropout = hp.Float('recurrent_dropout', 0.0, 0.5, default=0.2)
    num_layers = hp.Int('num_layers', 1, 3, default=1)

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
                input_shape=(X_train.shape[1], X_train.shape[2])
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


random_tuner = RandomSearch(
    hypermodel=build,
    objective='val_mse',
    max_trials=30,
    executions_per_trial=3,     
    directory=r'C:\keras_tuning',
    project_name='kerastuner_random_search',
    overwrite=True
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

random_tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    callbacks=callbacks,
    verbose=1
)


best_hp_rs = random_tuner.get_best_hyperparameters(1)[0]
print("Best RandomSearch HPs:")
for k, v in best_hp_rs.values.items():
    print(f"{k}: {v}")

best_model_rs = random_tuner.get_best_models(num_models=1)[0]
best_model_rs.summary()


#Best RandomSearch HPs:
#activation: relu
#recurrent_dropout: 0.35396854431693076
#num_layers: 1
#units_lstm_1: 16
#learning_rate: 0.01
#units_lstm_2: 176
#units_lstm_3: 64
#Model: "sequential"
#┌─────────────────────────────────┬────────────────────────┬───────────────┐
#│ Layer (type)                    │ Output Shape           │       Param # │
#├─────────────────────────────────┼────────────────────────┼───────────────┤
#│ lstm (LSTM)                     │ (None, 16)             │         1,152 │
#├─────────────────────────────────┼────────────────────────┼───────────────┤
#│ dense (Dense)                   │ (None, 1)              │            17 │
#└─────────────────────────────────┴────────────────────────┴───────────────┘
#Total params: 1,169 (4.57 KB)
#Trainable params: 1,169 (4.57 KB)
#Non-trainable params: 0 (0.00 B)








##### mais uma tentativa tuner


# RANDOM SEARCH (ajustado às suas novas considerações)
def build(hp):
    # hiperparâmetros conforme pedido
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, default=1)

    #units_choices = list(range(8, 65, 8))
    #units_choices = hp.Int('units_choices', values =[8,16,32])
    #units_choices = hp.Int('units_choices', 8, 16, 32)
    #units_choices = hp.Choice(f'units_lstm_{i+1}', [8, 16, 32])

    # dropout após o(s) LSTM(s)
    dropout_rate = hp.Choice('dropout', values=[0.0, 0.1, 0.2, 0.3, 0.4])

    # ativação da Dense final
    dense_activation = hp.Choice('dense_activation', values=['relu', 'linear', 'softplus'])

    # learning rate
    lr = hp.Choice('learning_rate', values=[1e-3, 1e-4])

    model = Sequential()

    for i in range(num_layers):
        #units_i = hp.Choice(f'units_lstm_{i+1}', values=units_choices)
        units_i = hp.Choice(f'units_lstm_{i+1}', [8, 16, 32])
        return_seq = (i < num_layers - 1)

        if i == 0:
            model.add(LSTM(
                units=units_i,
                activation='tanh',             # fixo
                return_sequences=return_seq,
                input_shape=(X_train.shape[1], X_train.shape[2])
            ))
        else:
            model.add(LSTM(
                units=units_i,
                activation='tanh',             # fixo
                return_sequences=return_seq
            ))

    # Dropout APÓS o bloco LSTM (uma única camada)
    model.add(Dropout(rate=dropout_rate))

    # Dense final (1 neurônio) com ativação testada
    model.add(Dense(1, activation=dense_activation))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mse',
        metrics=['mse']
    )
    return model

random_tuner = RandomSearch(
    hypermodel=build,
    objective='val_loss',
    max_trials=30,
    executions_per_trial=3,     
    directory=r'C:\keras_tuning',
    project_name='kerastuner_random_search',
    overwrite=True
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    #keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

random_tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=callbacks,
    verbose=1
)


best_hp_rs = random_tuner.get_best_hyperparameters(1)[0]
print("Best RandomSearch HPs:")
for k, v in best_hp_rs.values.items():
    print(f"{k}: {v}")

best_model_rs = random_tuner.get_best_models(num_models=1)[0]
best_model_rs.summary()

'''Best RandomSearch HPs: segunda tent
num_layers: 1
dropout: 0.4
dense_activation: softplus
learning_rate: 0.001
units_lstm_1: 8
units_lstm_2: 8
units_lstm_3: 16
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm (LSTM)                     │ (None, 8)              │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 8)              │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 1)              │             9 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 329 (1.29 KB)
 Trainable params: 329 (1.29 KB)
 Non-trainable params: 0 (0.00 B)
'''

''' primeira tent
Best RandomSearch HPs:
num_layers: 3
dropout: 0.3
dense_activation: softplus
learning_rate: 0.001
units_lstm_1: 40
units_lstm_2: 8
units_lstm_3: 8
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm (LSTM)                     │ (None, 22, 40)         │         6,720 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_1 (LSTM)                   │ (None, 22, 8)          │         1,568 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_2 (LSTM)                   │ (None, 8)              │           544 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 8)              │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 1)              │             9 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 8,841 (34.54 KB)
 Trainable params: 8,841 (34.54 KB)
 Non-trainable params: 0 (0.00 B)
'''
