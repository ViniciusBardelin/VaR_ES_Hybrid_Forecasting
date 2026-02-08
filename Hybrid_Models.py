import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense

os.environ['TF_DETERMINISTIC_OPS'] = '1'
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ———————————————————————— #
# —————— GARCH-LSTM —————— #
# ———————————————————————— #

df = pd.read_csv("ins_data.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
df['RV_lag1'] = df['RV_APPLE'].shift(1)
df['RV_lag5'] = df['RV_APPLE'].shift(5)
df[['RV_lag1','RV_lag5']] = df[['RV_lag1','RV_lag5']].bfill()
df = df.sort_values('Date').reset_index(drop=True)

window_size   = 22
initial_train = 2500
retrain_every = 252
feature_cols = ['Sigma2_GARCH']   
target_col   = 'RV_APPLE'         

features = df[feature_cols].values.astype(np.float32)      
target   = df[[target_col]].values.astype(np.float32)      
dates    = df['Date']
N = len(df)

scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.fit(target[:initial_train])
scaled_target = scaler_y.transform(target).flatten()

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_X.fit(features[:initial_train])
scaled_features = scaler_X.transform(features)             

def make_windows(X_arr, y_arr, size):
    X, y = [], []
    for i in range(size, len(X_arr)):
        X.append(X_arr[i-size:i, :])      
        y.append(y_arr[i])                
    X = np.array(X, dtype=np.float32)     )
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y

val_frac = 0.10
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

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
rlr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

def build_model(input_shape):
    model = keras.models.Sequential([
        LSTM(
            units=16,
            activation='relu',
            recurrent_dropout=0.35396854431693076,
            input_shape=input_shape
        ),
        Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mse']
    )
    return model

# Initial train
model = build_model((window_size, len(feature_cols)))
history = model.fit(
    X_tr, y_tr,
    epochs=100,
    batch_size=32,
    shuffle=False,
    validation_data=(X_val, y_val),
    callbacks=[es, rlr],
    verbose=1
)

# In-sample residuals
eps = 1e-12
returns_col = 'Returns'   

X_ins, _ = make_windows(
    scaled_features[:initial_train],
    scaled_target[:initial_train],
    window_size
)

p_ins = model.predict(X_ins, verbose=0) # (2478, 1)
rv_hat_ins = scaler_y.inverse_transform(p_ins)[:, 0] # (2478,)
rv_hat_ins = np.maximum(rv_hat_ins, eps)
sigma_hat_ins = np.sqrt(rv_hat_ins) # (2478,)

returns_window = df[returns_col].values[:initial_train] # (2500,)
mu_ins = returns_window.mean()

returns_ins = df[returns_col].values[window_size:initial_train] # (2478,)


resid_ins = (returns_ins - mu_ins) / sigma_hat_ins # (2478,)

dates_ins = df['Date'].iloc[window_size:initial_train].reset_index(drop=True)


plt.figure(figsize=(12,4))
plt.plot(dates_ins, resid_ins, linewidth=0.8)
plt.title("Resíduos padronizados in-sample (GARCH-LSTM)")
plt.xlabel("Data")
plt.ylabel("Resíduo")
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
plt.tight_layout()
plt.show()

dates_ins = df['Date'].iloc[window_size:initial_train].reset_index(drop=True)

os.makedirs("Res", exist_ok=True)
pd.DataFrame({"Date": dates_ins, "Residual": resid_ins}).to_csv(
    "Res/GARCH_LSTM_residuals.csv", index=False
)

# Walk-forward OoS
preds, pred_dates = [], []

for t in range(initial_train, N):

    # re-treina incrementalmente a cada bloco
    if (t - initial_train) % retrain_every == 0:

       
        scaler_X.fit(features[:t])
        scaler_y.fit(target[:t])  

        scaled_features[:t] = scaler_X.transform(features[:t]).astype(np.float32)
        scaled_target[:t]   = scaler_y.transform(target[:t]).flatten().astype(np.float32)

       
        X_new, y_new = make_windows(
            scaled_features[:t],
            scaled_target[:t],
            window_size
        )  

        model.fit(
            X_new, y_new,
            epochs=100,
            batch_size=32,
            shuffle=False,
            validation_split=0.1,   
            callbacks=[es, rlr],
            verbose=1
        )

   
    window = scaled_features[t-window_size:t]  
    x_in   = window.reshape(1, window_size, len(feature_cols))

    p_scaled = model.predict(x_in, verbose=0)                 
    p_orig   = scaler_y.inverse_transform(p_scaled)[0, 0]     

    preds.append(float(p_orig))
    pred_dates.append(dates.iloc[t])

df_pred = pd.DataFrame({
    'Date':       pred_dates,
    'Prediction': preds   
})

# Final plot
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['RV_APPLE'],
         label='RV_APPLE', color='tab:blue', linewidth=1, alpha=0.5)
plt.plot(df_pred['Date'], df_pred['Prediction'],
         label='GARCH-LSTM', color='tab:green', linewidth=2)
plt.title("Volatility Forecasts - GARCH-LSTM", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Volatility", fontsize=12)
plt.legend(loc="upper right", frameon=True, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
#plt.savefig("Resultados/GARCH_LSTM_T101_tst_carlos1.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Save CSV
df_pred.to_csv("DF_PREDS/GARCH_LSTM_T101_tst_carlos_1.csv", index=False)


# VaR/ES  
sigma2_col  = "Sigma2_GARCH"
sigma2_oos = df_pred["Prediction"].to_numpy(dtype=float) # len = n_oos
pred_dates = pd.to_datetime(df_pred["Date"]).reset_index(drop=True)

returns = df[returns_col].to_numpy(dtype=float) # len = N total (ins+oos)
sigma2_series = df[sigma2_col].to_numpy(dtype=float) # len = N total (variância GARCH para cada t)

n_oos = len(sigma2_oos)

returns = df[returns_col].to_numpy(float)
sigma2_oos = df_pred["Prediction"].to_numpy(float)  # variância prevista t+1 (len = n_oos)
pred_dates = pd.to_datetime(df_pred["Date"]).reset_index(drop=True)

q_1 = np.quantile(resid_ins, 0.01)
q_5 = np.quantile(resid_ins, 0.05)

n_oos = len(sigma2_oos)

VaR_1 = np.empty(n_oos)
VaR_5 = np.empty(n_oos)
ES_1  = np.empty(n_oos)
ES_5  = np.empty(n_oos)
r_oos = np.empty(n_oos)

for i in range(n_oos):
    returns_window = returns[i:i+n_ins]
    mu = returns_window.mean()

    sigma_t = max(sigma2_oos[i], 1e-12) # variância prevista
    sig = np.sqrt(sigma_t) # desvio padrão previsto

    VaR_1[i] = mu + sig * q_1
    VaR_5[i] = mu + sig * q_5

    ES_1[i] = returns_window[returns_window < VaR_1[i]].mean()
    ES_5[i] = returns_window[returns_window < VaR_5[i]].mean()

    r_oos[i] = returns[i + n_ins]

df_oos = pd.DataFrame({
    "Date": pred_dates,
    "Return": r_oos,
    "VaR_1": VaR_1,
    "VaR_5": VaR_5,
    "ES_1": ES_1,
    "ES_5": ES_5
})


hit_1 = (df_oos["Return"] < df_oos["VaR_1"]).mean()
hit_5 = (df_oos["Return"] < df_oos["VaR_5"]).mean()

print("Check VaR 1%:", hit_1, "|", (df_oos["Return"] < df_oos["VaR_1"]).sum(), "/", n)
print("Check VaR 5%:", hit_5, "|", (df_oos["Return"] < df_oos["VaR_5"]).sum(), "/", n)


# ————— Gráfico final —————
plt.figure(figsize=(14, 6))
plt.plot(df_oos['Date'], df_oos['VaR_1'],
         label='VaR 1%', color='tab:blue', linewidth=1, alpha=0.5)
plt.plot(df_oos['Date'], df_oos['Return'],
         label='Returns',color='tab:red', linewidth=1, alpha=0.5)
#plt.plot(df_pred['Date'], df_pred['Prediction'],
#         label='GARCH-LSTM', color='tab:green', linewidth=2)
plt.title("VaR 1% - GARCH-LSTM", fontsize=14)
plt.xlabel("Data", fontsize=12)
plt.ylabel("VaR", fontsize=12)
plt.legend(loc="upper right", frameon=True, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
#plt.savefig("Resultados/GARCH_LSTM_T101_tst_carlos1.pdf", format="pdf", bbox_inches="tight")
plt.show()