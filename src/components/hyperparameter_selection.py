from src.utils import data_label_split, save_object
from dataclasses import dataclass
import json
from src.logger import logging
from src.exception import CustomException
from src.components import data_transformation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class HyperparameterConfig:
    tuning_result_values_save_path=f'artifacts/tuning_results/{list(models.keys())[i]}.pkl'
    tuning_result_image_save_path=f'artifacts/tuning_results/{list(models.keys())[i]}.png'
    
class HyperparameterTuning(self):

    def __init__(self):
        self.hyperparameter_config=HyperparameterConfig()

    def hyperparameter_selection(X_train_path,models_param_path,cv=3):
        try:
            x,y = data_label_split(X_train_path)

            with open(models_param_path, 'r') as fp:
                  models = json.load(fp)

            for i in range(len(list(models))):

                logging.info(f"reading model parameters file.")
                model = list(models.keys())[i]
                model = getattr(sys.modules[__name__], model)()
                para = list(models.values())[i]

                transform = data_transformation.DataTransformation.get_data_transformer_object('transform')

                pipe = make_pipeline( transform , model )

                try:
                  logging.info(f"performing gridsearch.")
                  clf = GridSearchCV( pipe, para, cv=3, return_train_score=True, refit=False,scoring='r2', verbose=0) 
                  clf.fit(x,y)
                  n_param = len(list(list(para.values()))[0])
                  plot_x = np.arange(n_param)
                except:
                  logging.info(f"performing bayessearch.")
                  clf=BayesSearchCV(pipe, para , n_iter=10, cv=3, verbose=0,return_train_score=True,refit=False,random_state=55,scoring='r2')
                  clf.fit(x,y)
                  plot_x=np.arange(10)

                save_object(self.hyperparameter_config.tuning_result_values_save_path, clf.cv_results_)

                logging.info(f"plotting hyperparameter results.")
                sns.set_style("whitegrid")
                plt.figure(figsize=(12,7))
                sns.lineplot(x=plot_x, y=clf.cv_results_['mean_train_score'], label='mean train r2 score')
                sns.lineplot(x=plot_x, y=clf.cv_results_['mean_test_score'], label='mean cv r2 score')
                plt.xticks(plot_x)
                plt.tick_params(labelright=True)
                plt.xlabel('gridsearch result index')
                plt.ylabel('mean train and cv r2 score')
                plt.title('Train & CV curve for hyperparameter selection')
                plt.legend()
                plt.savefig(self.hyperparameter_config.tuning_result_image_save_path)
                plt.close()
                print(f'results for {list(models.keys())[i]} saved in artifacts folder for further evaluation.')

        except Exception as e:
            raise CustomException(e, sys)
