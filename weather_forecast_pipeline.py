"""
Pipeline object that bundles preprocessors and models for easy saving/loading and inference.
"""
import joblib
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import timedelta

FEATURE_COLS = ['_tempm','_hum','_pressurem','_wspdm']
DATETIME_COL = 'datetime_utc'
DEFAULT_HORIZON = 168


def refine_weather_logic(row):
    """Example rule-based refinement. Modify as needed.
    Return None to use model prediction, or return a string label override.
    """
    # Simple heuristic example: if humidity > 90 and wind < 3 -> 'Fog'
    try:
        if row['_hum'] > 90 and row['_wspdm'] < 3:
            return 'Fog'
    except Exception:
        return None
    return None


class WeatherForecastPipeline:
    def __init__(self, imputer, scaler, lstm, xgb_clf, label_encoder, seq_len=168):
        self.imputer = imputer
        self.scaler = scaler
        self.lstm = lstm
        self.xgb_clf = xgb_clf
        self.label_encoder = label_encoder
        self.seq_len = seq_len

    def predict_next_week(self, df_raw, horizon=DEFAULT_HORIZON):
        df_local = df_raw.sort_values(DATETIME_COL).reset_index(drop=True).copy()
        recent = df_local.tail(self.seq_len).copy()
        recent[FEATURE_COLS] = self.imputer.transform(recent[FEATURE_COLS])
        recent_scaled = self.scaler.transform(recent[FEATURE_COLS])
        X_input = recent_scaled.reshape((1, self.seq_len, len(FEATURE_COLS)))

        # LSTM prediction (scaled)
        y_pred_scaled = self.lstm.predict(X_input)[0]
        y_pred_scaled = y_pred_scaled[:horizon]

        # inverse scale
        y_pred = self.scaler.inverse_transform(y_pred_scaled)

        last_time = pd.to_datetime(df_local[DATETIME_COL].iloc[-1])
        times = [last_time + timedelta(hours=i+1) for i in range(horizon)]

        pred_df = pd.DataFrame(y_pred, columns=FEATURE_COLS)
        pred_df[DATETIME_COL] = times
        pred_df['hour'] = pred_df[DATETIME_COL].dt.hour
        pred_df['weekday'] = pred_df[DATETIME_COL].dt.weekday

        # XGBoost prediction
        X_for_xgb = pred_df[FEATURE_COLS + ['hour', 'weekday']]
        weather_enc = self.xgb_clf.predict(X_for_xgb)
        weather_pred = self.label_encoder.inverse_transform(weather_enc)
        pred_df['weather_model'] = weather_pred

        # refine with rules
        refined = []
        for _, row in pred_df.iterrows():
            rule = refine_weather_logic(row)
            refined.append(rule if rule is not None else row['weather_model'])
        pred_df['weather'] = refined
        return pred_df

    def save(self, path):
        base = os.path.splitext(path)[0]
        keras_path = base + '_lstm.keras'
        # Save keras model
        if self.lstm is not None:
            self.lstm.save(keras_path)
        # Temporarily remove keras model for joblib
        lstm_model = self.lstm
        self.lstm = None
        joblib.dump(self, path)
        # restore
        self.lstm = lstm_model
        return path

    @staticmethod
    def load(path):
        base = os.path.splitext(path)[0]
        keras_path = base + '_lstm.keras'
        obj = joblib.load(path)
        # load keras model
        obj.lstm = tf.keras.models.load_model(keras_path)
        return obj