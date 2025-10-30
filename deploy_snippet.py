import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow

# --- CONFIG ---
BUCKET = "weather-mlops-main"                      # your S3 bucket
DATA_KEY = "train_with_weather.csv"                # FIXED: dataset at root
OUTPUT_PATH = f"s3://{BUCKET}/output/"             # where model artifacts will go
ROLE = "arn:aws:iam::898423169134:role/service-role/AmazonSageMaker-ExecutionRole-20251028T094309"  # correct execution role

# --- SageMaker Session ---
sess = sagemaker.Session()

# --- S3 paths ---
train_input = f"s3://{BUCKET}/{DATA_KEY}"

# --- Define TensorFlow Estimator (runs your train.py) ---
estimator = TensorFlow(
    entry_point="train.py",
    source_dir=".",                # train.py + weather_forecast_pipeline.py here
    role=ROLE,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version="2.14",      # TensorFlow version
    py_version="py310",            # Python version
    output_path=OUTPUT_PATH,       # trained model artifacts go here automatically
    base_job_name="weather-forecast-train",
    hyperparameters={"epochs": 6}
)

print("Estimator defined")

# --- Launch training job ---
estimator.fit({"train": train_input}, wait=True)

print("Estimator trained")

# --- Deploy model as endpoint ---
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    entry_point="inference.py",    # inference script to handle requests
    source_dir=".",                # inference.py + pipeline file
)

print("✅ Deployment complete!")
print("Endpoint name:", predictor.endpoint_name)
