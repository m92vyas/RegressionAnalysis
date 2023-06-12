from src.utils import data_label_split, load_object, save_object
from src.logger import logging
from src.exception import CustomException
from src.components import data_transformation
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import sys
from dataclasses import dataclass


@dataclass
class ModelEvaluationConfig:
    model_save_path=f'artifacts/saved_models/{str(model)[:10]}.pkl'
    r2_score_save_path=f'artifacts/r2_scores/{str(model)[:10]}.pkl'

class ModelEvaluation(self):

  def __init__(self):
    self.modelevaluationconfig=ModelEvaluationConfig()

  def model_evaluation(X_train_path, X_test_path, model_path, param_index, cv_results_path, save_model=False):
    try:
      x,y = data_label_split(X_train_path)
      x_test,y_test = data_label_split(X_test_path)
      cv_results = load_object(cv_results_path)
      model = load_object(model_path)
      para = cv_results['params'][param_index]
      transform = data_transformation.DataTransformation.get_data_transformer_object('transform')
      pipe = make_pipeline( transform , model )
      pipe.set_params(**para)
      pipe.fit(x,y)
      y_predict_train = pipe.predict(x)
      y_predict_test = pipe.predict(x_test)
      r2_train= r2_score(y, y_predict_train)
      r2_test= r2_score(y_test, y_predict_test)
      save_object(self.modelevaluationconfig.r2_score_save_path,(r2_train,r2_test))

      if save_model==True:
        save_object(self.modelevaluationconfig.model_save_path,pipe)
      return r2_train,r2_test

    except Exception as e:
      raise CustomException(e, sys)
