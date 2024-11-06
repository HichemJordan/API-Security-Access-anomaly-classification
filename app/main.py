# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load("../models/hgbt_final.joblib")


# Define the request format
class APICallFeatures(BaseModel):
    max_global_source_degrees: int
    avg_global_source_degrees: float
    min_global_dest_degrees: int
    std_local_source_degrees: float
    max_global_dest_degrees: int
    min_global_source_degrees: int
    std_global_source_degrees: float
    n_connections: int
    avg_global_dest_degrees: float


# Initialize the FastAPI app
app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "API Security Classification Service"}


@app.post("/predict")
def predict(features: APICallFeatures):
    try:
        # Convert input data into the format your model expects
        data = np.array(
            [
                [
                    features.max_global_source_degrees,
                    features.avg_global_source_degrees,
                    features.min_global_dest_degrees,
                    features.std_local_source_degrees,
                    features.max_global_dest_degrees,
                    features.min_global_source_degrees,
                    features.std_global_source_degrees,
                    features.n_connections,
                    features.avg_global_dest_degrees,
                ]
            ]
        )
        # Make a prediction
        prediction = model.predict(data)
        # Return whether the API call is anomalous (1) or normal (0)
        return {"anomalous": bool(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
