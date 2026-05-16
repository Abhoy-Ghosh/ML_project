# =========================================================
# IMPORT REQUIRED LIBRARIES
# =========================================================

import os
import sys

from dataclasses import dataclass

# Custom logger
from src.logger import logging

# Custom exception handler
from src.exception import CustomException

# Utility functions
from src.utils import save_object
from src.utils import evaluate_models

# =========================================================
# MACHINE LEARNING MODELS
# =========================================================

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Evaluation metric
from sklearn.metrics import r2_score


# =========================================================
# CONFIGURATION CLASS
# =========================================================

@dataclass
class ModelTrainerConfig:

    # Path where trained model will be saved

    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


# =========================================================
# MODEL TRAINER CLASS
# =========================================================

class ModelTrainer:

    # Constructor

    def __init__(self):

        # Create configuration object

        self.model_trainer_config = ModelTrainerConfig()


    # =====================================================
    # MAIN MODEL TRAINING FUNCTION
    # =====================================================

    def initiate_model_trainer(self, train_array, test_array):

        '''
        This function is responsible for:

        1. Splitting features and target
        2. Training multiple ML models
        3. Hyperparameter tuning
        4. Selecting best model
        5. Saving best trained model
        6. Returning final R2 score
        '''

        try:

            logging.info("Splitting training and testing input data")


            # =================================================
            # SPLIT INPUT FEATURES AND TARGET COLUMN
            # =================================================

            # train_array structure:
            #
            # [features | target]
            #
            # :-1  -> all columns except last
            # -1   -> last column (target)

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]


            # =================================================
            # DEFINE MODELS
            # =================================================

            models = {

                "Linear Regression": LinearRegression(),

                "Lasso Regression": Lasso(),

                "Ridge Regression": Ridge(),

                "K-Neighbors Regressor": KNeighborsRegressor(),

                "Decision Tree Regressor": DecisionTreeRegressor(),

                "Random Forest Regressor": RandomForestRegressor(),

                "XGBRegressor": XGBRegressor(),

                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False
                ),

                "AdaBoost Regressor": AdaBoostRegressor(),

                "Gradient Boosting Regressor": GradientBoostingRegressor()
            }


            # =================================================
            # HYPERPARAMETERS FOR GRIDSEARCHCV
            # =================================================

            params = {

                "Linear Regression": {},

                "Lasso Regression": {
                    'alpha': [0.1, 0.01, 1]
                },

                "Ridge Regression": {
                    'alpha': [0.1, 0.01, 1]
                },

                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 10]
                },

                "Decision Tree Regressor": {
                    'criterion': [
                        'squared_error',
                        'friedman_mse',
                        'absolute_error',
                        'poisson'
                    ]
                },

                "Random Forest Regressor": {
                    'n_estimators': [8, 16, 32, 64]
                },

                "Gradient Boosting Regressor": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [8, 16, 32]
                },

                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [8, 16, 32]
                },

                "CatBoosting Regressor": {
                    'depth': [6, 8],
                    'iterations': [30, 50]
                },

                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [8, 16, 32]
                }
            }


            # =================================================
            # TRAIN MODELS + GET SCORES
            # =================================================

            # model_report:
            #
            # {
            #   model_name : r2_score
            # }

            # trained_models:
            #
            # {
            #   model_name : fitted_model
            # }

            model_report, trained_models = evaluate_models(

                X_train=X_train,
                y_train=y_train,

                X_test=X_test,
                y_test=y_test,

                models=models,
                param=params
            )


            # =================================================
            # GET BEST MODEL SCORE
            # =================================================

            best_model_score = max(model_report.values())


            # =================================================
            # GET BEST MODEL NAME
            # =================================================

            best_model_name = max(
                model_report,
                key=model_report.get
            )


            # =================================================
            # GET BEST TRAINED MODEL
            # =================================================

            best_model = trained_models[best_model_name]


            logging.info(
                f"Best model found: "
                f"{best_model_name} "
                f"with score: {best_model_score}"
            )


            # =================================================
            # VALIDATE MODEL PERFORMANCE
            # =================================================

            if best_model_score < 0.6:

                logging.info(
                    "No best model found"
                )

                raise CustomException(
                    "No best model found",
                    sys
                )


            # =================================================
            # SAVE BEST MODEL
            # =================================================

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model
            )


            logging.info(
                "Best trained model saved successfully"
            )


            # =================================================
            # FINAL PREDICTION
            # =================================================

            y_predicted = best_model.predict(X_test)


            # =================================================
            # FINAL R2 SCORE
            # =================================================

            r2_square = r2_score(
                y_test,
                y_predicted
            )


            logging.info(
                f"Final R2 Score: {r2_square}"
            )


            # =================================================
            # RETURN FINAL SCORE
            # =================================================

            return r2_square


        # =====================================================
        # EXCEPTION HANDLING
        # =====================================================

        except Exception as e:

            raise CustomException(e, sys)


# =========================================================
# COMPLETE FLOW
# =========================================================

"""
                MODEL TRAINING FLOW

          Transformed Train/Test Arrays
                        ↓
              Split Features & Target
                        ↓
                Initialize Models
                        ↓
               Hyperparameter Tuning
                  (GridSearchCV)
                        ↓
                 Train All Models
                        ↓
                 Evaluate R2 Score
                        ↓
                 Compare All Models
                        ↓
                 Select Best Model
                        ↓
                 Save Best Model
                        ↓
                  Final Prediction
                        ↓
                   Return R2 Score


=========================================================

WHY THIS FILE EXISTS?

Purpose of Model Trainer:

1. Centralized training logic
2. Compare multiple models
3. Tune hyperparameters
4. Select best model automatically
5. Save trained model
6. Keep pipeline modular

=========================================================

IMPORTANT ENGINEERING CONCEPTS USED

1. Modular Architecture
2. OOP Design
3. GridSearchCV
4. Hyperparameter Tuning
5. Model Serialization
6. Pipeline Engineering
7. Logging
8. Exception Handling

=========================================================
"""