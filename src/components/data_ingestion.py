import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_excel('daily_offers.xlsx')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")

            df = df.drop(columns=['id','item_date','customer','country','item type','material_ref'])
            df = df[df['selling_price']<2000]
            df = df[df['selling_price']>0]
            # getting rows where quantity is not float and -ve     
            df = df.reset_index(drop=True)
            def index_notfloat(x):
              index_qty_notfloat = []
              for i in range(len(x)):
                try:
                  float(x['quantity tons'].values[i])
                except:
                  index_qty_notfloat.append(i)
              return index_qty_notfloat

            indx = index_notfloat(df)
            df = df.drop(indx , axis=0)
            df = df.dropna()
            df = df.reset_index(drop=True)

            train_set,test_set=train_test_split(df,test_size=0.1,random_state=92)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)