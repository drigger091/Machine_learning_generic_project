import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object

@dataclass

class model_trainerConfig:

    trained_model_path = os.path.join("datasets","model.pkl")

class modeltrainer:
    def __init__(self):
 
        self.model_trainer_config =model_trainerConfig()


        def initiate_model_trainer(self,train_array,test_array,preprocessor_path):

            try:
                logging.info("Split the train and test data")

                X_train ,Y_train ,X_test , Y_test = (train_array[:,:-1],
                                                     train_array[:,-1],
                                                     test_array[:, :-1],
                                                     test_array[:, -1])
                
                models = {
                    "Linear Regression" :  LinearRegression(),
                    'Decision Tree'     :   DecisionTreeRegressor(),
                    'K Nearest Neighbors':    KNeighborsRegressor(),
                    'Gradient Boosting': GradientBoostingRegressor(),
                    'Adaboost Regressor': AdaBoostRegressor(),
                    'CatBoost Regressor': CatBoostRegressor(verbose=False),
                     'xgboost regressor':      XGBRegressor(),
                     'Random Forest Regressor':RandomForestRegressor()
                     }
                
                model_report:dict = evaluate_model(X_train=X_train,Y_train=Y_train,X_test = X_test,Y_test=Y_test,models = models)
                                                                        
            except:
                pass



