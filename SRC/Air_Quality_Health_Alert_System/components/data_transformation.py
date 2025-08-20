import pandas as pd
import os
import numpy as np
from pathlib import Path
from Air_Quality_Health_Alert_System import logger
from Air_Quality_Health_Alert_System.entity.config_entity  import DataTransformationConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import joblib
import warnings
warnings.filterwarnings('ignore')


class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.imputers = {}
        
    def load_and_preprocess_data(self):
       
        logger.info("Loading and preprocessing data...")
        
        
        df = pd.read_csv(self.config.data_path, parse_dates=['date'])
        logger.info(f"Loaded data shape: {df.shape}")
        
       
        df = df.sort_values('date')
        
        
        df = df.infer_objects()
        
        return df
    
    def handle_missing_values(self, df):
        
        df = df.copy()
        logger.info("Handling missing values...")
        
        
        missing_before = df.isnull().sum().sum()
        logger.info(f"Missing values before processing: {missing_before}")
        
       
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
     
        if 'date' in categorical_cols:
            categorical_cols.remove('date')
        
      
        if numeric_cols:
            # For time series data, use time-based interpolation first
            df_with_date = df.set_index('date') if 'date' in df.columns else df
            df_with_date[numeric_cols] = df_with_date[numeric_cols].interpolate(method='time')
            df = df_with_date.reset_index() if 'date' in df_with_date.index.names else df_with_date
            
            
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        
        if categorical_cols:
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values after processing: {missing_after}")
        
        return df
    
    def create_advanced_features(self, df):
        
        df = df.copy()
        logger.info("Creating advanced features...")
        
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Temporal features (if not already present)
            if 'month' not in df.columns:
                df['month'] = df['date'].dt.month
            if 'day' not in df.columns:
                df['day'] = df['date'].dt.day
            if 'year' not in df.columns:
                df['year'] = df['date'].dt.year
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['date'].dt.dayofweek
            if 'hour' not in df.columns and df['date'].dt.hour.nunique() > 1:
                df['hour'] = df['date'].dt.hour
            if 'is_weekend' not in df.columns:
                df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
            if 'day_of_year' not in df.columns:
                df['day_of_year'] = df['date'].dt.dayofyear
            
            
            df['quarter'] = df['date'].dt.quarter
            df['week_of_year'] = df['date'].dt.isocalendar().week
            
            
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            if 'hour' in df.columns:
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # Seasonal features
            df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
            df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
            df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
            df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)
        
        
        pollutant_cols = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
        
        
        if 'pm25' in df.columns and 'pm10' in df.columns:
            df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 1e-6)
            df['pm_total'] = df['pm25'] + df['pm10']
        
        if 'no2' in df.columns and 'co' in df.columns:
            df['no2_co_ratio'] = df['no2'] / (df['co'] + 1e-6)
        
        
        if 'wind_speed_mps' in df.columns:
            for pollutant in pollutant_cols:
                if pollutant in df.columns:
                    df[f'{pollutant}_wind_interaction'] = df[pollutant] * df['wind_speed_mps']
        
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['lat_lon_interaction'] = df['latitude'] * df['longitude']
           
        
       
        if 'city' in df.columns:
            
            if hasattr(self.config, 'target_column') and self.config.target_column in df.columns:
                city_means = df.groupby('city')[self.config.target_column].mean()
                df['city_target_encoded'] = df['city'].map(city_means)
            
            
            if df['city'].nunique() <= 10:
                city_dummies = pd.get_dummies(df['city'], prefix='city', drop_first=True)
                df = pd.concat([df, city_dummies], axis=1)
        
        logger.info(f"Features created. New shape: {df.shape}")
        return df
    
    def handle_outliers(self, df, method='iqr', factor=3.0):
        
        df = df.copy()
        logger.info("Handling outliers...")
        
        
        pollutant_cols = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
        weather_cols = ['wind_speed_mps']
        
        outlier_cols = [col for col in pollutant_cols + weather_cols if col in df.columns]
        
        initial_shape = df.shape[0]
        
        if method == 'iqr':
            for col in outlier_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                # Remove outliers
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            for col in outlier_cols:
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < factor]
        
        final_shape = df.shape[0]
        removed = initial_shape - final_shape
        logger.info(f"Outliers removed: {removed} rows ({removed/initial_shape*100:.2f}%)")
        
        return df
    
    def scale_features(self, df, method='minmax'):
        
        df = df.copy()
        logger.info(f"Scaling features using {method} scaler...")
        
      
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
       
        exclude_cols = [
            'latitude', 'longitude', 'month', 'day', 'year', 'day_of_week', 
            'hour', 'is_weekend', 'day_of_year', 'quarter', 'week_of_year',
            'is_winter', 'is_summer', 'is_spring', 'is_autumn',
            
            getattr(self.config, 'target_column', None),
           
        ]
        
        
        exclude_cols = [col for col in exclude_cols if col is not None and col in df.columns]
        
        
        city_cols = [col for col in df.columns if col.startswith('city_') and col != 'city_target_encoded']
        exclude_cols.extend(city_cols)
        
        
        features_to_scale = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Columns to scale: {len(features_to_scale)}")
        logger.info(f"Excluded columns: {len(exclude_cols)}")
        
        if features_to_scale:
            # Choose scaler
            if method == 'minmax':
                self.scaler = MinMaxScaler()
            elif method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            # Fit and transform
            df[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])
            logger.info(f"Scaled {len(features_to_scale)} features")
        
        return df
    
     
    def train_test_splitting(self):
        
        logger.info("Starting comprehensive data transformation pipeline...")
        
        
        df = self.load_and_preprocess_data()
        
                
       
        df = self.handle_missing_values(df)
        
        
        df = self.create_advanced_features(df)
        
        
        if hasattr(self.config, 'remove_outliers') and self.config.remove_outliers:
            df = self.handle_outliers(df, 
                                    method=getattr(self.config, 'outlier_method', 'iqr'),
                                    factor=getattr(self.config, 'outlier_factor', 3.0))
        
        
        scaling_method = getattr(self.config, 'scaling_method', 'minmax')
        df = self.scale_features(df, method=scaling_method)
        
       
        
        
       
        logger.info("Performing train-test split...")
        
       
        df = df.sort_values('date') if 'date' in df.columns else df
        
        
        split_ratio = getattr(self.config, 'train_split_ratio', 0.75)
        split_index = int(len(df) * split_ratio)
        
        train = df.iloc[:split_index].copy()
        test = df.iloc[split_index:].copy()
        
        
        if 'date' in train.columns:
            train = train.reset_index(drop=True)
            test = test.reset_index(drop=True)
        
       
        os.makedirs(self.config.root_dir, exist_ok=True)
        
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        
        
        if self.scaler is not None:
            scaler_path = os.path.join(self.config.root_dir, "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")
        
   
        logger.info("=== Train-Test Split Summary ===")
        logger.info(f"Total samples: {len(df):,}")
        logger.info(f"Training samples: {len(train):,} ({len(train)/len(df)*100:.1f}%)")
        logger.info(f"Test samples: {len(test):,} ({len(test)/len(df)*100:.1f}%)")
        logger.info(f"Features: {train.shape[1]}")
        logger.info(f"Train date range: {train['date'].min()} to {train['date'].max()}" if 'date' in train.columns else "No date column")
        logger.info(f"Test date range: {test['date'].min()} to {test['date'].max()}" if 'date' in test.columns else "No date column")
        
        
        if hasattr(self.config, 'target_column') and self.config.target_column in train.columns:
            numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
            if self.config.target_column in numeric_features:
                numeric_features.remove(self.config.target_column)
            
            if len(numeric_features) > 0:
                correlations = train[numeric_features + [self.config.target_column]].corr()[self.config.target_column].abs().sort_values(ascending=False)[1:11]
                logger.info("Top 10 features by correlation with target:")
                for feature, corr in correlations.items():
                    logger.info(f"  {feature}: {corr:.3f}")
        
        logger.info("Data transformation pipeline completed successfully!")
        
       
        return train, test