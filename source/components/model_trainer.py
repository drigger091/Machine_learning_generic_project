import os
import sys

from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import ( AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from source.logger import logging
from source.exception import CustomException

from source.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join("Datasets","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):

        try:

            logging.info("Split Training and test input data")

            X_train ,Y_train ,X_test ,Y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )

            models ={
                "RandomForest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "KNN" :  KNeighborsRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'CatBoost' : CatBoostRegressor(),
                "XGB":XGBRegressor(),
                "Linear":LinearRegression(),
            }

            model_report:dict = evaluate_models(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models)

        ## derving the best score from dict
            best_model_score = max(sorted(model_report.values()))

        # to get the best model from the dict

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:

                raise CustomException("No best Model Found")
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted = best_model.predict(X_test)

            r2_Score = r2_score(Y_test,predicted)

            return r2_Score
    

        except Exception as e:
            raise CustomException(e,sys)
