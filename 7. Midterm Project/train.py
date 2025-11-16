import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline


def load_data():
    df = pd.read_csv("data.csv")

    df.drop("Unnamed: 0", axis=1, inplace=True)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # lowercase categorical strings
    categorical = list(df.dtypes[df.dtypes == "object"].index)
    for c in categorical:
        df[c] = df[c].str.lower().str.replace(" ", "_")

    return df


def train_model(df):
    categorical = ['song_title', 'artist']

    # REMOVE target from features
    numerical = [
        col for col in df.columns if col not in categorical and col != "target"]

    y = df.target.values
    train_dict = df[categorical + numerical].to_dict(orient="records")

    pipeline = make_pipeline(
        DictVectorizer(),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    )

    pipeline.fit(train_dict, y)
    return pipeline


def save_model(pipeline, output_file):
    with open(output_file, "wb") as f_out:
        pickle.dump(pipeline, f_out)


df = load_data()
pipeline = train_model(df)
save_model(pipeline, "model.bin")

print("Model saved to model.bin")
