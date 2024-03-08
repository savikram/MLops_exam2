import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from Diabetes_model import __version__ as _version
from Diabetes_model.config.core import config
from Diabetes_model.pipeline import Diabetes_pipe
from Diabetes_model.processing.data_manager import load_pipeline
from Diabetes_model.processing.data_manager import pre_pipeline_preparation
#from Diabetes_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
Diabetes_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    #validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.models_cfg.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = Diabetes_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = Diabetes_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    data_in= {'Pregnancies':[9], 'Glucose': [120],'BloodPressure':[72],'SkinThickness':[22],'Insulin':[56],'BMI':[20.8],'DiabetesPedigreeFunction':[0.7333],'Age':[48]}
    
    
    predictions = Diabetes_pipe.predict(pd.DataFrame(data_in))
    print(predictions)
    #make_prediction(input_data=data_in)
