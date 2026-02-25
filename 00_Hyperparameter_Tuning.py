import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

eps = 1e-12
lookback = 22
val_frac = 0.20

target_col  = "RV_APPLE"
returns_col = "Returns"  

feature_cols = ["Sigma2_GARCH", "RV_lag1", "RV_lag5", "AbsRet", "Ret2"]

df = pd.read_csv("ins_data.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df = df.sort_values("Date").reset_index(drop=True)

N_INS = 2500
df = df.iloc[:N_INS].copy()

df["RV_lag1"] = df[target_col].shift(1)
df["RV_lag5"] = df[target_col].shift(5)
df[["RV_lag1", "RV_lag5"]] = df[["RV_lag1", "RV_lag5"]].bfill()

df["AbsRet"] = df[returns_col].abs()
df["Ret2"]   = df[returns_col] ** 2

features = df[feature_cols].values.astype(np.float32)
target_rv = df[target_col].values.astype(np.float32)

log_target = np.log(np.maximum(target_rv, eps)).astype(np.float32)  

N = len(df)
val_start = int(N * (1 - val_frac))

X_train_raw = features[:val_start]
y_train_log = log_target[:val_start]

X_val_raw   = features[val_start - lookback:]
y_val_log   = log_target[val_start - lookback:]

x_scaler = MinMaxScaler(feature_range=(0, 1))
X_train = x_scaler.fit_transform(X_train_raw).astype(np.float32)
X_val   = x_scaler.transform(X_val_raw).astype(np.float32)

Y_MEAN = y_train_log.mean().astype(np.float32)
Y_STD  = (y_train_log.std() + 1e-8).astype(np.float32)  

y_train_scaled = ((y_train_log - Y_MEAN) / Y_STD).astype(np.float32)
y_val_scaled   = ((y_val_log   - Y_MEAN) / Y_STD).astype(np.float32)

def make_windows(x_2d, y_1d, lookback: int):
    X, y = [], []
    for t in range(lookback, len(x_2d)):
        X.append(x_2d[t - lookback:t, :])
        y.append(y_1d[t])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y

X_tr, y_tr = make_windows(X_train, y_train_scaled, lookback)
X_va, y_va = make_windows(X_val,   y_val_scaled,   lookback)

print("X_tr:", X_tr.shape, "y_tr:", y_tr.shape)
print("X_va:", X_va.shape, "y_va:", y_va.shape)

Y_MEAN_T = tf.constant(Y_MEAN, dtype=tf.float32)
Y_STD_T  = tf.constant(Y_STD,  dtype=tf.float32)
EPS_T    = tf.constant(eps,    dtype=tf.float32)

@tf.function
def qlike_loss_from_scaled_log(y_true_scaled, y_pred_scaled):
    y_true_log = y_true_scaled * Y_STD_T + Y_MEAN_T
    y_pred_log = y_pred_scaled * Y_STD_T + Y_MEAN_T

    rv_true = tf.exp(tf.clip_by_value(y_true_log, -50.0, 50.0))
    rv_hat  = tf.exp(tf.clip_by_value(y_pred_log, -50.0, 50.0))

    rv_hat = tf.maximum(rv_hat, EPS_T)

    loss = rv_true / rv_hat + tf.math.log(rv_hat)
    return tf.reduce_mean(loss)

def build(hp):
    num_layers = hp.Int("num_layers", 1, 3, default=1)
    dropout_rate = hp.Choice("dropout", [0.0, 0.1, 0.2, 0.3, 0.4])
    lr = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])

    model = Sequential()

    for i in range(num_layers):
        units_i = hp.Choice(f"units_lstm_{i+1}", [8, 16, 32, 64])
        return_seq = (i < num_layers - 1)

        if i == 0:
            model.add(LSTM(
                units=units_i,
                activation="tanh",
                return_sequences=return_seq,
                input_shape=(X_tr.shape[1], X_tr.shape[2])
            ))
        else:
            model.add(LSTM(
                units=units_i,
                activation="tanh",
                return_sequences=return_seq
            ))

    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss=qlike_loss_from_scaled_log,
        metrics=[qlike_loss_from_scaled_log]
    )
    return model

class MyRandomTuner(kt.tuners.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Choice("batch_size", [16, 32])
        kwargs["shuffle"] = False  
        return super().run_trial(trial, *args, **kwargs)

tuner = MyRandomTuner(
    hypermodel=build,
    objective=kt.Objective("val_loss", direction="min"),
    max_trials=80,
    executions_per_trial=3,
    directory=r"C:\keras_tuning",
    project_name="rs_qlike_logrv",
    overwrite=True
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

tuner.search(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("\nBest HPs:")
for k, v in best_hp.values.items():
    print(f"{k}: {v}")

best_model = tuner.get_best_models(1)[0]
best_model.summary()