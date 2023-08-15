import os
import sys
from typing import Annotated

import joblib
import pandas as pd
from classifier.classifier import ModelClassifier
from fastapi import FastAPI, Path
from sklearn.model_selection import train_test_split
from starlette.responses import JSONResponse
from train.train_data import FraudDetectionPipeline, Retrieve_Files

from .models.models import OnlineTX

# refactored folder
REFACTORED_DIRECTORY = "/Users/francisco.torres/Documents/GitHub/MLOps_project/Refactor/mlops_project/mlops_project"
DATASETS_DIR = "./data/"  # Directory where data will be unzip.
RETRIEVED_DATA = (
    "retrieved_data.csv"  # File name for the retrieved data without irrelevant columns
)

TARGET = "isFraud"
FEATURES = ["type", "amount", "oldbalanceOrg", "newbalanceOrig"]
NUMERICAL_VARS = ["amount", "oldbalanceOrg", "newbalanceOrig"]
CATEGORICAL_VARS = ["type"]
NUMERICAL_VARS_WITH_NA = []
CATEGORICAL_VARS_WITH_NA = []


SELECTED_FEATURES = [
    "type_CASH_OUT",
    "type_PAYMENT",
    "type_CASH_IN",
    "type_TRANSFER",
    "type_rare",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
]
SELECTED_FEATURES = ["type", "amount", "oldbalanceOrg", "newbalanceOrig"]

SEED_MODEL = 725
# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

PIPELINE_NAME = "DecisionTree"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output.pkl"
MODEL_DIRECTORY = "/Users/francisco.torres/Documents/GitHub/MLOps_project/Refactor/mlops_project/mlops_project/models/"
TRAINED_MODEL_DIR = "./models/"
PIPELINE_NAME = "DecisionTree"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output.pkl"

# Persist/Save model
SAVE_FILE_NAME = f"{PIPELINE_SAVE_FILE}"
SAVE_PATH = TRAINED_MODEL_DIR + SAVE_FILE_NAME


app = FastAPI()


@app.get("/", status_code=200)
async def healthcheck():
    return "Online Fraud Classifier is all ready to go!"


@app.post("/classify")
def classify(Online_TX_features: OnlineTX):
    predictor = ModelClassifier(MODEL_DIRECTORY + PIPELINE_SAVE_FILE)
    X = [
        Online_TX_features.type,
        Online_TX_features.amount,
        Online_TX_features.oldbalanceOrg,
        Online_TX_features.newbalanceOrg,
    ]
    prediction = predictor.predict([X])
    return JSONResponse(f"Resultado predicción: {prediction}")


@app.get("/train_model", status_code=200)
def train_model():
    # Change location to the refactored directory
    os.chdir(REFACTORED_DIRECTORY)

    # This class will retrieve ZIP file and extract csv
    retrieve_files = Retrieve_Files()
    result = retrieve_files.retrieve_files()
    # Instantiate the FraudDetectionPipeline class
    fraud_data_pipeline = FraudDetectionPipeline(
        seed_model=SEED_MODEL,
        numerical_vars=NUMERICAL_VARS,
        categorical_vars_with_na=CATEGORICAL_VARS_WITH_NA,
        numerical_vars_with_na=NUMERICAL_VARS_WITH_NA,
        categorical_vars=CATEGORICAL_VARS,
        selected_features=SELECTED_FEATURES,
    )

    # Read data
    df = pd.read_csv(DATASETS_DIR + RETRIEVED_DATA)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(TARGET, axis=1), df[TARGET], test_size=0.2, random_state=3
    )

    # Fit the model

    DecisionTreeModel = fraud_data_pipeline.fit_DecisionTree(X_train, y_train)

    result = joblib.dump(DecisionTreeModel, SAVE_PATH)
    print(f"Model saved in {result}")

    return "Trained model ready to go!"
