# Airbnb MLOps
Курсовой проект ML/DLOps.
Данные: Inside Airbnb.
Задача семинара: чтение CSV из S3.


## Technologies

- Python 3.11
- Conda
- JupyterLab
- Pandas
- Boto3
- Git

## Environment Setup
Conda environment named mlops26 with Python 3.11.
Activate the environment: conda activate mlops26
Install the required packages: pip install pandas boto3 jupyterlab ipykernel

## Data Source
from boto3:
Endpoint: https://storage.yandexcloud.net
Bucket: mlops26-course-bucket
Object key: prepared/train.csv

from local CSV (data/) if s3 is not available

## S3 Credentials
S# access credentials is in secretkeys.py