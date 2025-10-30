import pickle

import pandas as pd
import numpy as np
import sklearn


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


def load_data():
    df = pd.read_csv(
        r"C:\Users\GIA DAT\ML Zoomcamp\5. Deployment\Homework 5\course_lead_scoring.csv")

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    return df


def train_model(df):
    categorical = ['lead_source']
    numeric = ['number_of_courses_viewed', 'annual_income']

    df[categorical] = df[categorical].fillna('NA')
    df[numeric] = df[numeric].fillna(0)

    y_train = df.converted
    train_dict = df[categorical + numeric].to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(),
        LogisticRegression(solver='liblinear')
    )

    pipeline.fit(train_dict, y_train)
    return pipeline


def save_model(pipeline, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)


df = load_data()
pipeline = train_model(df)
save_model(pipeline, 'model.bin')

print('Model saved to model.bin')
