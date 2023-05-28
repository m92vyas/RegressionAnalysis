import os
import sys
import pickle
from src.exception import CustomException
import pandas as pd

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def data_label_split(data_path):
  try:
    data = pd.read_csv(data_path)
    y=data.selling_price
    x=data.drop(columns=['selling_price'])
    return x,y
  
  except Exception as e:
    raise CustomException(e, sys)