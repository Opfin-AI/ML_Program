# your_model_module.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

SEQ_LENGTH = 60


def build_model(n_features):
    inp = Input(shape=(SEQ_LENGTH, n_features))
    x = LSTM(50)(inp)
    x = Dropout(0.2)(x)
    r = Dense(1, name='reg_output')(x)
    b = Dense(1, activation='sigmoid', name='bin_output')(x)
    model = Model(inputs=inp, outputs=[r, b])
    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss={'reg_output': 'mse', 'bin_output': 'binary_crossentropy'},
        loss_weights={'reg_output': 0.5, 'bin_output': 0.5}
    )
    model.make_predict_function()
    return model


def run_model_for_streamlit(df, timeframe):
    df = df.copy()
    if 'timestamp' not in df.columns:
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})
        else:
            raise KeyError("DataFrame must contain either 'timestamp' or 'Date' column")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    df['return_1'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))

    df['target_reg'] = df['Close'].shift(-1) / df['Close'] - 1.0
    df['target_bin'] = (df['target_reg'] > 0).astype(int)
    df.dropna(subset=['target_reg', 'target_bin'], inplace=True)

    features = df[['return_1', 'SMA_10', 'SMA_20', 'RSI_14']].ffill().bfill()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features.values.astype('float32'))

    X, y_reg, y_bin = [], [], []
    for i in range(SEQ_LENGTH, len(X_scaled)):
        X.append(X_scaled[i-SEQ_LENGTH:i])
        y_reg.append(df['target_reg'].iloc[i])
        y_bin.append(df['target_bin'].iloc[i])

    X = np.array(X)
    y_reg = np.array(y_reg)
    y_bin = np.array(y_bin)

    model = build_model(X.shape[2])
    model.fit(X, {'reg_output': y_reg, 'bin_output': y_bin}, epochs=3, batch_size=32, verbose=0)

    preds_reg, preds_bin = model.predict(X, batch_size=32, verbose=0)
    preds_reg = preds_reg.flatten()
    preds_bin = (preds_bin.flatten() > 0.5).astype(int)

    timestamps = df.index[SEQ_LENGTH:]
    true_prices = df['Close'].iloc[SEQ_LENGTH:] * (1 + y_reg)
    pred_prices = df['Close'].iloc[SEQ_LENGTH:] * (1 + preds_reg)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timestamps, true_prices, label='Actual Price (t+1)', color='C0')
    ax.plot(timestamps, pred_prices, label='Predicted Price (t+1)', color='C3')
    ax.set_title(f"{timeframe.upper()} Price Prediction")
    ax.legend()
    ax.grid(True)

    metrics = {
        'mae': mean_absolute_error(y_reg, preds_reg) * 100,
        'rmse': np.sqrt(mean_squared_error(y_reg, preds_reg)) * 100,
        'dir_acc': accuracy_score(y_bin, preds_bin) * 100
    }

    predictions_df = pd.DataFrame({
        'timestamp': timestamps,
        'actual_close': df['Close'].iloc[SEQ_LENGTH:],
        'predicted_close': pred_prices,
        'predicted_return': preds_reg,
        'predicted_direction': preds_bin
    })

    return fig, predictions_df, metrics
