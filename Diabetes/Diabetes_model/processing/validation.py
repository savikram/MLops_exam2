import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from Diabetes_model.config.core import config
from Diabetes_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(df=input_df)
    validated_data = pre_processed[config.models_cfg.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    Pregnancies:Optional[int]
    Glucose: Optional[int]
    BloodPressure:Optional[int]
    SkinThickness:Optional[int]
    Insulin:Optional[int]
    BMI:Optional[float]
    DiabetesPedigreeFunction:Optional[float]
    Age:Optional[int]
    


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]