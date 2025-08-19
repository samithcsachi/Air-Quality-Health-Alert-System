import pandas as pd
import os
import numpy as np
from Air_Quality_Health_Alert_System import logger
from Air_Quality_Health_Alert_System.entity.config_entity  import DataTransformationConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def train_test_spliting(self):
        df = pd.read_csv(self.config.data_path, parse_dates=['date'])  
        df = df.sort_values('date')  

        df.set_index('date', inplace=True) 

        # Impute missing values using interpolation
        df.interpolate(method='time', inplace=True)

        # Select features for modeling
        numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        exclude_cols = ['latitude', 'longitude', 'aqi', 'month', 'day', 'year', 'day_of_week', 
                'hour', 'is_weekend', 'day_of_year']  # keep original
        features_to_scale = [col for col in numeric_features if col not in exclude_cols]

        

        scaler = MinMaxScaler()
        for col in features_to_scale:
            df[col] = df[col].fillna(df[col].mean())

        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

            
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(df,test_size = 0.25, random_state = 42)



        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splitted, cleaned, and over sampled data into training and test sets.")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")

        print("Train:", train.shape)
        print("Test:", test.shape)
