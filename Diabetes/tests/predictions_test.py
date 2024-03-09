import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from Diabetes_model.predict import predict_score


def test_prediction_score(sample_input_data):
    # Given
    #print(type(sample_input_data))
    expected_no_predictions = 308       

    result = predict_score(input_data=sample_input_data[0], label_data=sample_input_data[1])
    
    # Then
    score = result.get("Accuracy")    
    assert result.get("errors") is None
    #assert len(predictions) == expected_no_predictions
    #_predictions = list(predictions)
    #print(len(_predictions))
    #y_true = sample_input_data[1]
    #accuracy = accuracy_score(_predictions, y_true)
    print("Accuracy:",score)
    assert score > 0.8
