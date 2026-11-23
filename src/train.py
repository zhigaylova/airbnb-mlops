
from pathlib import Path
import os

import boto3
import joblib
import numpy as np
import pandas as pd

from botocore.exceptions import BotoCoreError, ClientError

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Paths and S3 configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

S3_BUCKET = "mlops26-course-bucket"
S3_KEY = "prepared/train.csv"

LOCAL_PATH = PROJECT_ROOT / "data" / "listing.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"


s3 = boto3.client(
    "s3",
    endpoint_url="https://storage.yandexcloud.net",
    aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
    region_name="ru-central1",
)


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def read_s3_csv(bucket, key):
    response = s3.get_object(
        Bucket=bucket,
        Key=key
    )

    df = pd.read_csv(response["Body"])
    return df


def read_data(bucket, key, local_path=None):
    try:
        df = read_s3_csv(bucket, key)
        return df

    except (BotoCoreError, ClientError, OSError) as e:
        if local_path is None:
            raise

        print(f"S3 error: {type(e).__name__}: {e}")
        print(f"Reading local file: {local_path}")

        df = pd.read_csv(local_path)
        return df


# --------------------------------------------------
# Data preparation
# --------------------------------------------------

def prepare_data(df):
    X = df.drop(
        ["listing_id", "snapshot_date", "price"],
        axis=1
    )

    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
        test_size=0.25,
        shuffle=True
    )

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# Pipeline creation
# --------------------------------------------------

def create_pipeline(X_train):

    num_cols = X_train.select_dtypes(
        include="number"
    ).columns

    cat_cols = X_train.select_dtypes(
        exclude="number"
    ).columns

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                ),
                cat_cols
            )
        ],
        remainder="passthrough"
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            LinearRegression(
                fit_intercept=True,
                n_jobs=-1
            )
        )
    ])

    return pipe


# --------------------------------------------------
# Training and evaluation
# --------------------------------------------------

def train_model(pipe, X_train, y_train):
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)

    rmse = root_mean_squared_error(
        y_test,
        y_pred
    )

    print("RMSE:", np.round(rmse, 4))


# --------------------------------------------------
# Model saving
# --------------------------------------------------

def save_model(pipe):
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(pipe, MODEL_PATH)

    print(f"Pipeline saved to: {MODEL_PATH}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    df = read_data(
        S3_BUCKET,
        S3_KEY,
        LOCAL_PATH
    )

    print("Dataset shape:", df.shape)

    X_train, X_test, y_train, y_test = prepare_data(df)

    pipe = create_pipeline(X_train)

    pipe = train_model(
        pipe,
        X_train,
        y_train
    )

    evaluate_model(
        pipe,
        X_test,
        y_test
    )

    save_model(pipe)


if __name__ == "__main__":
    main()
