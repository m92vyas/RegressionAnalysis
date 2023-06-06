from src.utils import load_object
import os
import sys
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","XGBRegress.pkl")
            model=load_object(file_path=model_path)
            preds=model.predict(features)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


def features_to_array(quantity_tons: float,
                      status: str,
                      application: float,
                      thickness: float,
                      width: float,
                      product_ref: int,
                      delivery_date: float):
  
  try:
    return np.array([[quantity_tons, status,	application,	thickness,	width,	product_ref,	delivery_date	]])
  except Exception as e:
    raise CustomException(e, sys)