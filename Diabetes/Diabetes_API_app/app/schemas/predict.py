from typing import Any, List, Optional

from pydantic import BaseModel
from Diabetes_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        'Pregnancies':3,
                        'Glucose': 158,
                        'BloodPressure':70,
                        'SkinThickness':30,
                        'Insulin':328,
                        'BMI':35.5,
                        'DiabetesPedigreeFunction':0.344,
                        'Age':35
                        
                    }
                ]
            }
        }
