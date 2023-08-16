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

            params = {
                  "Decision Tree":{
                      'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                      #'splitter':['best','random'],
                      #'max_features':['sqrt','log2'],
                  },
                  "RandomForest":{
                      #'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                      #'max_features':['sqrt','log2',None],
                      'n_estimators':[8,16,32,64,128,256]

                  },
                   "KNN": {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  },
                  "Gradient Boosting":{
                        "learning_rate":[.1,.01,.05,.001],
                        #'loss':['squared_error','huber','absolute_error','quantile'],
                        'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                        #'criterion':['squared_error','friedman_mse'],
                        #'max_features':['auto','sqrt','log2'],
                        'n_estimators':[8,16,32,64,128,256]
                   },
                   "XGB": {
                        'learning_rate': [0.1, 0.01, 0.05, 0.001],
                        #'max_depth': [3, 4, 5, 6, 7, 8, 9],
                        #'min_child_weight': [1, 2, 3, 4],
                        #'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                        #'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                        'n_estimators': [8, 16, 32, 64, 128, 256]

                   },
                   "Linear": {
                   },
                   "CatBoost": {
                       'iterations' : [30,50,100],
                       'learning_rate':[0.01,0.05,0.1],
                       'depth':[6,8,10]

                   },
                   "AdaBoost":{
                       'learning_rate': [.1,.01,.05,.001],
                       #'loss':['linear','square','exponential'],
                       'n_estimators':[8,16,32,64,128,256]
                   }
            }

            model_report:dict = evaluate_models(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models,param = params)

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
