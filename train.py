"""
Entry point for SageMaker training job.
Saves artifacts to the directory given by --output-dir or SM_MODEL_DIR.
"""
import argparse
import os
import json
import joblib
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import xgboost as xgb
import tensorflow as tf

from weather_forecast_pipeline import WeatherForecastPipeline, refine_weather_logic, FEATURE_COLS, DATETIME_COL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--seq-len', type=int, default=168)
    parser.add_argument('--horizon', type=int, default=168)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--random-seed', type=int, default=42)
    return parser.parse_args()


def supervised_sequences(values, seq_len=168, horizon=168):
    X, y = [], []
    n_total = len(values)
    for start in range(0, n_total - seq_len - horizon + 1):
        X.append(values[start:start+seq_len])
        y.append(values[start+seq_len:start+seq_len+horizon])
    return np.asarray(X), np.asarray(y)


def build_lstm_seq2seq(n_features, seq_len=168, horizon=168, units=64):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

    encoder_inputs = Input(shape=(seq_len, n_features))
    encoder = LSTM(units, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    decoder_inputs = RepeatVector(horizon)(state_h)
    decoder_lstm = LSTM(units, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(n_features))
    outputs = decoder_dense(decoder_outputs)
    model = Model(encoder_inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.random_seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Expect training CSV named train_with_weather.csv inside train channel
    train_csv_path = os.path.join(args.train_data, 'train_with_weather.csv')
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training CSV not found at {train_csv_path}")

    print('Loading data from', train_csv_path)
    df = pd.read_csv(train_csv_path)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)

    # Split out final year for holdout (keeps behavior similar to your notebook)
    last_date = df[DATETIME_COL].max()
    one_year_ago = last_date - pd.DateOffset(years=1)
    df_last_year_test = df[df[DATETIME_COL] > one_year_ago].copy()
    df = df[df[DATETIME_COL] <= one_year_ago].copy()

    # Preprocessing
    imputer = SimpleImputer(strategy='mean')
    df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS])
    joblib.dump(imputer, os.path.join(args.output_dir, 'imputer.joblib'))

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[FEATURE_COLS] = scaler.fit_transform(df_scaled[FEATURE_COLS])
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.joblib'))

    df["hour"] = pd.to_datetime(df[DATETIME_COL]).dt.hour
    df["weekday"] = pd.to_datetime(df[DATETIME_COL]).dt.weekday

    # Sequence data for LSTM
    values = df_scaled[FEATURE_COLS].values
    X_seq, y_seq = supervised_sequences(values, seq_len=args.seq_len, horizon=args.horizon)
    print('Sequence shapes:', X_seq.shape, y_seq.shape)

    split_idx = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    # Build and train LSTM
    n_features = len(FEATURE_COLS)
    lstm = build_lstm_seq2seq(n_features, seq_len=args.seq_len, horizon=args.horizon, units=64)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]

    lstm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save LSTM model separately
    lstm_path = os.path.join(args.output_dir, 'lstm.keras')
    lstm.save(lstm_path)
    print('Saved LSTM to', lstm_path)

    # Train XGBoost classifier for weather label
    X_for_xgb = df[FEATURE_COLS + ['hour', 'weekday']]
    y_for_xgb = df['weather'].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y_for_xgb)
    joblib.dump(le, os.path.join(args.output_dir, 'label_encoder.joblib'))

    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
        X_for_xgb, y_enc, test_size=0.2, random_state=args.random_seed, stratify=y_enc
    )

    xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
    xgb_clf.fit(X_train_xgb, y_train_xgb, eval_set=[(X_test_xgb, y_test_xgb)], verbose=False)

    joblib.dump(xgb_clf, os.path.join(args.output_dir, 'xgb_clf.joblib'))
    print('Saved XGBoost to', os.path.join(args.output_dir, 'xgb_clf.joblib'))

    # Create and save unified pipeline object
    pipeline = WeatherForecastPipeline(
        imputer=imputer,
        scaler=scaler,
        lstm=lstm,
        xgb_clf=xgb_clf,
        label_encoder=le,
        seq_len=args.seq_len
    )

    pipeline_path = os.path.join(args.output_dir, 'weather_forecast_pipeline.pkl')
    pipeline.save(pipeline_path)
    print('Saved pipeline to', pipeline_path)

    # Save a small metadata file
    meta = {
        'n_seq_samples': len(X_seq),
        'seq_len': args.seq_len,
        'horizon': args.horizon
    }
    with open(os.path.join(args.output_dir, 'training_metadata.json'), 'w') as f:
        json.dump(meta, f)

    print('Training completed successfully. Artifacts saved to', args.output_dir)

## Usage notes

# 1. Put `train_with_weather.csv` inside the SageMaker training channel (or upload to S3 and pass as channel). The script expects the file name `train_with_weather.csv` in the training channel folder.
# 2. `train.py` writes artifacts to `/opt/ml/model` (SageMaker default). The `pipeline.save()` writes `weather_forecast_pipeline.pkl` and `weather_forecast_pipeline_lstm.keras` (suffix `_lstm.keras`).
# 3. `inference.py` loads the pipeline and returns JSON with `predictions` list.
# 4. Adjust `refine_weather_logic()` in `weather_forecast_pipeline.py` to include any domain rules you need.

# ---

# If you'd like, I can also:

# * produce a single `model.tar.gz` packaging step required for custom containers,
# * generate a Dockerfile for a custom inference image,
# * create a small unit-test notebook that runs locally with the included code.

# Tell me which of those you want next.
