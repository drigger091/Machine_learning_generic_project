import os
import sys
import numpy as np
import pandas as pd
from source.exception import CustomException
import dill


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    


def evaluate_model(X_train,Y_train,X_test,Y_test,models):

    try:

        report ={}

        for i in range(len(list(models))):

            model = list(models.values())[i]

            model.fit(X_train,Y_train)  # train model

            Y

