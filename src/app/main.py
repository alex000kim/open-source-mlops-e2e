import json
import sys
from pathlib import Path

import uvicorn

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

from typing import List

import pandas as pd
from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel

from utils.load_params import load_params

app = FastAPI()
# https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

params = load_params(params_path='params.yaml')
model_path = params.train.model_path
feat_cols = params.base.feat_cols
model = load(filename=model_path)

class Customer(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class Request(BaseModel):
    data: List[Customer]

@app.post("/predict")
async def predict(info: Request = Body(..., example={
    "data": [
        {
            "CreditScore": 619,
            "Age": 42,
            "Tenure": 2,
            "Balance": 0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 101348.88
        },
        {
            "CreditScore": 699,
            "Age": 39,
            "Tenure": 21,
            "Balance": 0,
            "NumOfProducts": 2,
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "EstimatedSalary": 93826.63
        }
    ]
})):
    json_list = json.loads(info.json())
    data = json_list['data']
    input_data = pd.DataFrame(data)
    probs = model.predict_proba(input_data)[:,0]
    probs = probs.tolist()
    return probs

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)