import os
import sys
import pandas as pd
import numpy as np
import dill
import  pickle


from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from dataclasses import dataclass


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):

    try:

        report = {}
        trained_models = {}

        for model_name, model in models.items():

            model_param = param[model_name]

            gs = GridSearchCV(
                estimator=model,
                param_grid=model_param,
                cv=3
            )

            gs.fit(X_train, y_train)

            # Best trained model
            best_model = gs.best_estimator_

            # Predictions
            y_test_pred = best_model.predict(X_test)

            # Score
            test_model_score = r2_score(y_test, y_test_pred)

            # Store score
            report[model_name] = test_model_score

            # Store trained model
            trained_models[model_name] = best_model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)