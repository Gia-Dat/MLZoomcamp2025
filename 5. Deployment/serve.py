# serve.py
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import uvicorn

# ========== your schema ==========


class Customer(BaseModel):
    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int
    monthlycharges: float
    totalcharges: float


# ========== load model ==========
with open("model.bin", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/invocations")
def invoke(customer: Customer):
    prob = model.predict_proba([customer.model_dump()])[0][1]
    return {"churn_probability": float(prob), "churn": prob >= 0.5}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
