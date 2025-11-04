import argparse
import os
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
import sys


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.bin")
    with open(model_path, "rb") as f_in:
        model = pickle.load(f_in)
    return model


if __name__ == "__main__":

    print("[INFO] Extracting arguments...")
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--solver", type=str, default="liblinear")
    parser.add_argument("--C", type=float, default=1.0)

    # SageMaker default arguments (folders)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")


    args, _ = parser.parse_known_args()

    print("SKLearn Version:", sklearn.__version__)
    print()

    # ===================== READ DATA =====================
    print("[INFO] Loading data...")

    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    train_df.columns = train_df.columns.str.lower().str.replace(" ", "_")
    test_df.columns = test_df.columns.str.lower().str.replace(" ", "_")

    # Clean categorical columns
    cat_cols = list(train_df.dtypes[train_df.dtypes == "object"].index)
    for c in cat_cols:
        train_df[c] = train_df[c].str.lower().str.replace(" ", "_")
        test_df[c] = test_df[c].str.lower().str.replace(" ", "_")

    # Fix totalcharges numeric issue
    train_df.totalcharges = pd.to_numeric(train_df.totalcharges, errors="coerce").fillna(0)
    test_df.totalcharges = pd.to_numeric(test_df.totalcharges, errors="coerce").fillna(0)

    # Target
    # train_df.churn = (train_df.churn == "yes").astype(int)
    # test_df.churn = (test_df.churn == "yes").astype(int)

    # Features
    numerical = ['tenure', 'monthlycharges', 'totalcharges']
    categorical = [
        'gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice',
        'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
        'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
        'contract', 'paperlessbilling', 'paymentmethod'
    ]

    X_train = train_df[categorical + numerical].to_dict(orient="records")
    y_train = train_df.churn

    unique_classes, counts = np.unique(y_train, return_counts=True)
    print(f"[INFO] Classes in training data: {dict(zip(unique_classes, counts))}")

    if len(unique_classes) < 2:
        print("[ERROR] Training data must have at least 2 classes. Exiting...")
        sys.exit(1)

    X_test = test_df[categorical + numerical].to_dict(orient="records")
    y_test = test_df.churn

    print("[INFO] Training Logistic Regression model...")
    pipeline = make_pipeline(
        DictVectorizer(),
        LogisticRegression(solver=args.solver, C=args.C)
    )

    pipeline.fit(X_train, y_train)

    # ===================== SAVE MODEL =====================
    model_path = os.path.join(args.model_dir, "model.bin")
    with open(model_path, "wb") as f_out:
        pickle.dump(pipeline, f_out)

    print(f"[INFO] Model saved at {model_path}")

    # ===================== EVALUATION =====================
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print()
    print("===== TEST METRICS =====")
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
