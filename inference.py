"""
SageMaker inference script. Exposes model_fn, input_fn, predict_fn, output_fn expected by SageMaker.
"""
import os
import json
import pandas as pd
from weather_forecast_pipeline import WeatherForecastPipeline, DATETIME_COL


def model_fn(model_dir):
    # model_dir is /opt/ml/model inside container
    pipeline_path = os.path.join(model_dir, 'weather_forecast_pipeline.pkl')
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError('Pipeline artifact not found in model_dir: ' + model_dir)
    pipeline = WeatherForecastPipeline.load(pipeline_path)
    return pipeline


def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        payload = json.loads(request_body)
        # Accept either list-of-dicts or dict-of-lists forms
        if isinstance(payload, dict) and 'records' in payload:
            df = pd.DataFrame(payload['records'])
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
        elif isinstance(payload, dict):
            # dict-of-lists -> DataFrame
            df = pd.DataFrame(payload)
        else:
            raise ValueError('Unsupported JSON payload format')
        # Ensure datetime column exists
        if DATETIME_COL not in df.columns:
            raise ValueError(f"Input JSON must include '{DATETIME_COL}' column with timestamps")
        df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
        return df
    else:
        raise ValueError('Unsupported content type: ' + content_type)


def predict_fn(input_data, model):
    # input_data is a pandas DataFrame
    # model is a WeatherForecastPipeline instance
    # Default horizon can be passed via header or payload; here we read from payload if provided
    horizon = None
    # If user passed a 'horizon' column or single int in input_data.attrs
    if hasattr(input_data, 'attrs') and 'horizon' in input_data.attrs:
        horizon = int(input_data.attrs['horizon'])
    # Otherwise fallback
    if horizon is None:
        horizon = 168
    pred_df = model.predict_next_week(input_data, horizon=horizon)
    return pred_df


def output_fn(prediction, accept='application/json'):
    # prediction is a pandas DataFrame
    if accept == 'application/json':
        records = prediction.to_dict(orient='records')
        return json.dumps({'predictions': records}), accept
    else:
        raise ValueError('Unsupported accept type: ' + accept)