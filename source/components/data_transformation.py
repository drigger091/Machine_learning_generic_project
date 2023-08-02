import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object



class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('datasets',"preprocessor.pkl")

class Datatransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        """This function is responsible for data Transformation"""
        try :

            numerical_colummns = [ 'reading score','writing score']
            categorical_columns= ['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']


            numerical_pipeline = Pipeline(
            #numerical pipeline step 1 - imputation of missing values with mean value using Imputer
                steps =[
                         ("imputer",SimpleImputer(strategy="median")),
                        ('scaler', StandardScaler(with_mean=False))
                        ])
            

            categorical_pipeline = Pipeline(

                steps= [
                    ("imputer",SimpleImputer(strategy = "most_frequent"))
                    ,("onehotencoder",OneHotEncoder()),
                    ('Scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns:{numerical_colummns}")
            
            preprocessor = ColumnTransformer(
                [("numerical_pipeline",numerical_pipeline,numerical_colummns),
                 ("categorical_pipeline",categorical_pipeline,categorical_columns)]
            )


            return preprocessor
        
        except Exception as e:
             raise CustomException(e,sys)
            



    def initiate_data_transformation(self, train_path ,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df =  pd.read_csv(test_path) 

            logging.info("Read train and test data completed")


            logging.info("Obtaining preproceesing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math score"
            numerical_columns = [ 'reading score','writing score']

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying Preprocessing object on training dataframe and testing dataframe")


            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array,np.array(target_feature_test_df)]


            logging.info(f"Saved Preprocressing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_ob_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)