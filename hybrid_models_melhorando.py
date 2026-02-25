import os
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ----------------------------
# Reprodutibilidade
# ----------------------------
os.environ["TF_DETERMINISTIC_OPS"] = "1"
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ----------------------------
# Config
# ----------------------------
window_size   = 22
initial_train = 2500
retrain_every = 100

eps = 1e-12
returns_col = "Returns"
target_col  = "RV_APPLE"

# ----------------------------
# Load / features
# ----------------------------
df = pd.read_csv("ins_data.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df = df.sort_values("Date").reset_index(drop=True)

# Lags de RV
df["RV_lag1"] = df[target_col].shift(1)
df["RV_lag5"] = df[target_col].shift(5)
df[["RV_lag1", "RV_lag5"]] = df[["RV_lag1", "RV_lag5"]].bfill()

# Features extras (ajudam bastante para capturar choques)
df["AbsRet"] = df[returns_col].abs()
df["Ret2"]   = df[returns_col] ** 2

# Features do híbrido
feature_cols = ["Sigma2_GARCH", "RV_lag1", "RV_lag5", "AbsRet", "Ret2"]

features = df[feature_cols].values.astype(np.float32)             # (N, k)
target_rv = df[[target_col]].values.astype(np.float32).flatten()  # (N,)
dates = df["Date"]
N = len(df)

# ----------------------------
# Target: log(RV)
# ----------------------------
log_target = np.log(np.maximum(target_rv, eps)).astype(np.float32)  # (N,)

# ----------------------------
# Scaling (fixo no initial_train)
# - Mantemos scaler_y FIXO para loss QLIKE com parâmetros constantes
# ----------------------------
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_X.fit(features[:initial_train])
scaled_features = scaler_X.transform(features).astype(np.float32)

scaler_y = StandardScaler()
scaler_y.fit(log_target[:initial_train].reshape(-1, 1))
scaled_target = scaler_y.transform(log_target.reshape(-1, 1)).astype(np.float32).flatten()

# parâmetros fixos para a loss (constantes no treino)
Y_MEAN = float(scaler_y.mean_[0])
Y_STD  = float(scaler_y.scale_[0])

# ----------------------------
# Window maker
# ----------------------------
def make_windows(X_arr, y_arr, size):
    X, y = [], []
    for i in range(size, len(X_arr)):
        X.append(X_arr[i-size:i, :])
        y.append(y_arr[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y

# ----------------------------
# QLIKE loss (usando RV em escala original)
# modelo prevê y_hat = scaled_log_RV
# loss usa:
#   logRV_hat = y_hat * Y_STD + Y_MEAN
#   RV_hat = exp(logRV_hat)
#   RV_true = exp(logRV_true)
#   QLIKE = RV_true / RV_hat + log(RV_hat)
# ----------------------------
@tf.function
def qlike_loss_from_scaled_log(y_true_scaled, y_pred_scaled):
    # desfaz escala -> log(RV)
    y_true_log = y_true_scaled * Y_STD + Y_MEAN
    y_pred_log = y_pred_scaled * Y_STD + Y_MEAN

    rv_true = tf.exp(tf.clip_by_value(y_true_log, -50.0, 50.0))
    rv_hat  = tf.exp(tf.clip_by_value(y_pred_log, -50.0, 50.0))

    rv_hat = tf.maximum(rv_hat, tf.constant(eps, dtype=rv_hat.dtype))

    # QLIKE (forma comum para variância)
    loss = rv_true / rv_hat + tf.math.log(rv_hat)
    return tf.reduce_mean(loss)

# ----------------------------
# Train/Val split (como você já fazia)
# ----------------------------
val_frac = 0.20
val_start = int(initial_train * (1 - val_frac))

X_tr, y_tr = make_windows(
    scaled_features[:val_start],
    scaled_target[:val_start],
    window_size
)

X_val, y_val = make_windows(
    scaled_features[val_start - window_size:initial_train],
    scaled_target[val_start - window_size:initial_train],
    window_size
)

print("X_tr:", X_tr.shape, "y_tr:", y_tr.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)

es = EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True)

# ----------------------------
# Model
# ----------------------------
def build_model(input_shape):
    model = Sequential([
        LSTM(16, activation="tanh", return_sequences=True, input_shape=input_shape),
        LSTM(8,  activation="tanh", return_sequences=True),
        LSTM(8,  activation="tanh", return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="linear")   # saída = scaled_log_RV
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=qlike_loss_from_scaled_log,
        metrics=[qlike_loss_from_scaled_log]
    )
    return model

# ----------------------------
# Initial train
# ----------------------------
model = build_model((window_size, len(feature_cols)))
history = model.fit(
    X_tr, y_tr,
    epochs=100,
    batch_size=16,
    shuffle=False,
    validation_data=(X_val, y_val),
    callbacks=[es],
    verbose=1
)

loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])
epochs_ran = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs_ran, loss, label="train loss")
plt.plot(epochs_ran, val_loss, label="val loss")
plt.xlabel("Epoch")
plt.ylabel("QLIKE")
plt.title(f"Initial training (val_frac={val_frac}): QLIKE vs val_QLIKE")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------
# Helper: prediction -> RV (original)
# ----------------------------
def pred_scaledlog_to_rv(p_scaledlog):
    # p_scaledlog: (n,1) em scaled_log_RV
    p_log = scaler_y.inverse_transform(p_scaledlog)[:, 0]   # log(RV)
    rv = np.exp(np.clip(p_log, -50, 50))
    rv = np.maximum(rv, eps)
    return rv

# ----------------------------
# In-sample fitted (a partir de t=window_size)
# ----------------------------
X_ins, _ = make_windows(
    scaled_features[:initial_train],
    scaled_target[:initial_train],
    window_size
)

p_ins_scaledlog = model.predict(X_ins, verbose=0)        # (n_ins-window, 1)
rv_hat_ins = pred_scaledlog_to_rv(p_ins_scaledlog)       # (n_ins-window,)
sigma_hat_ins = np.sqrt(rv_hat_ins)

returns_window = df[returns_col].values[:initial_train].astype(float)
mu_ins = returns_window.mean()

returns_ins = df[returns_col].values[window_size:initial_train].astype(float)
resid_ins = (returns_ins - mu_ins) / sigma_hat_ins

dates_ins = df["Date"].iloc[window_size:initial_train].reset_index(drop=True)

plt.figure(figsize=(12, 4))
plt.plot(dates_ins, resid_ins, linewidth=0.8)
plt.title("Resíduos padronizados in-sample (GARCH-LSTM | logRV + QLIKE)")
plt.xlabel("Data")
plt.ylabel("Resíduo")
ax = plt.gca()
loc = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
plt.tight_layout()
plt.show()

# ----------------------------
# Walk-forward OoS (retrain) - mantendo scalers fixos
# ----------------------------
preds, pred_dates = [], []

for t in range(initial_train, N):

    # re-treina em blocos (expanding window), sem refit dos scalers
    if (t - initial_train) % retrain_every == 0:
        X_new, y_new = make_windows(
            scaled_features[:t],
            scaled_target[:t],
            window_size
        )

        model.fit(
            X_new, y_new,
            epochs=100,
            batch_size=16,
            shuffle=False,
            validation_split=0.1,
            callbacks=[es],
            verbose=1
        )

    window = scaled_features[t - window_size:t]
    x_in = window.reshape(1, window_size, len(feature_cols))

    p_scaledlog = model.predict(x_in, verbose=0)          # (1,1) scaled_log_RV
    p_rv = pred_scaledlog_to_rv(p_scaledlog)[0]           # RV_hat(t)

    preds.append(float(p_rv))
    pred_dates.append(dates.iloc[t])

df_pred = pd.DataFrame({"Date": pred_dates, "Prediction": preds})

# ----------------------------
# Plot RV_true vs RV_hat OoS
# ----------------------------
plt.figure(figsize=(14, 6))
plt.plot(df["Date"], df[target_col], label="RV_true", linewidth=1, alpha=0.5)
plt.plot(df_pred["Date"], df_pred["Prediction"], label="GARCH-LSTM (logRV+QLIKE)", linewidth=2)
plt.title("Volatility Forecasts - GARCH-LSTM (logRV + QLIKE)")
plt.xlabel("Date")
plt.ylabel("Realized Variance (RV)")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------
# DF FULL (fitted INS + forecast OoS)
# df_full começa em window_size (por construção do fitted)
# ----------------------------
df_ins = pd.DataFrame({
    "Date": dates_ins.values,
    "Returns": returns_ins.astype(float),
    "RV_true": df[target_col].iloc[window_size:initial_train].values.astype(float),
    "RV_hat": rv_hat_ins.astype(float),
    "Set": "INS"
})

returns_oos = df[returns_col].iloc[initial_train:].values.astype(float)

df_oos = pd.DataFrame({
    "Date": np.array(pred_dates),
    "Returns": returns_oos,
    "RV_true": df[target_col].iloc[initial_train:].values.astype(float),
    "RV_hat": np.array(preds, dtype=float),
    "Set": "OOS"
})

df_full = (
    pd.concat([df_ins, df_oos], axis=0, ignore_index=True)
      .sort_values("Date")
      .reset_index(drop=True)
)

df_full.to_csv("GARCH_LSTM_2.csv", index=False)
print("Saved:", "GARCH_LSTM_2.csv", "| shape:", df_full.shape)
