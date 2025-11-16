# predict.py
import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn


class Customer(BaseModel):
    # numerical
    acousticness: float = Field(..., ge=0.0, le=1.0)
    danceability: float = Field(..., ge=0.0, le=1.0)
    duration_ms: int = Field(..., ge=1)
    energy: float = Field(..., ge=0.0, le=1.0)
    instrumentalness: float = Field(..., ge=0.0, le=1.0)
    liveness: float = Field(..., ge=0.0, le=1.0)
    loudness: float
    speechiness: float = Field(..., ge=0.0, le=1.0)
    tempo: float
    valence: float = Field(..., ge=0.0, le=1.0)

    # categorical
    song_title: str
    artist: str

    # integer-coded categorical
    key: int = Field(..., ge=0, le=11)
    mode: int = Field(..., ge=0, le=1)
    time_signature: int = Field(..., ge=1, le=5)


class PredictResponse(BaseModel):
    like_prob: float
    target: bool


app = FastAPI(title="favorite-song-customer")


# load model
with open("model.bin", "rb") as f_in:
    model = pickle.load(f_in)


def predict_single(customer_dict):
    pred = model.predict_proba([customer_dict])[0, 1]
    return float(pred)


@app.post("/predict", response_model=PredictResponse)
def predict(customer: Customer):
    customer_dict = customer.model_dump()
    prob = predict_single(customer_dict)
    return PredictResponse(
        like_prob=prob,
        target=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
