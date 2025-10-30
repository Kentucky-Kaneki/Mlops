import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

role = "arn:aws:iam::898423169134:user/weather-mlops-jayaram"
sess = sagemaker.Session()

estimator = sagemaker.estimator.Estimator(
    entry_point='train.py',
    source_dir='.',
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    image_uri=None,  # let SageMaker pick a suitable image or use prebuilt framework estimator
    hyperparameters={'epochs': 6}
)

# Upload local training CSV to S3 and point to that channel. Example:
s3_input = sagemaker.Session().upload_data(path='train_with_weather.csv', key_prefix='weather-train')
estimator.fit({'train': s3_input})

# After training, deploy
predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m5.large')