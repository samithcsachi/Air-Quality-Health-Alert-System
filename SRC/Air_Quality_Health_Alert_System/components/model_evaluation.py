import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from Air_Quality_Health_Alert_System import logger
from Air_Quality_Health_Alert_System.entity.config_entity  import ModelEvaluationConfig
from pathlib import Path



class ModelEvaluation:
    def __init__(self, config):
        self.config = config
        self.predictions = None   
        self.actuals = None 

    def create_lag_features(self, df, target_col='aqi', lags=[1, 2, 3], rolling_windows=[3, 7]):
       
        df = df.copy()
        
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Creating lag features for {target_col}...")
        
       
        for lag in lags:
            df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
            
       
        for window in rolling_windows:
            df[f"{target_col}_rolling{window}"] = df[target_col].shift(1).rolling(window=window).mean()
            df[f"{target_col}_rolling{window}_std"] = df[target_col].shift(1).rolling(window=window).std()
        
 
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        logger.info(f"Lag features created. Rows: {initial_rows} -> {final_rows} (removed {initial_rows - final_rows} NaN rows)")
        logger.info(f"Lag features created: {[col for col in df.columns if 'lag' in col or 'rolling' in col]}")
        return df

    def create_seasonal_features(self, df):
        
        df = df.copy()
        
        if 'season' in df.columns:
            logger.info("Creating one-hot encoded seasonal features...")
           
            seasonal_dummies = pd.get_dummies(df['season'], prefix='season', drop_first=False)
            
           
            df = pd.concat([df, seasonal_dummies], axis=1)
            
            
            df = df.drop('season', axis=1)
            
            
            logger.info(f"Seasonal features created: {[col for col in seasonal_dummies.columns]}")
        else:
            logger.info("Warning: 'season' column not found!")
        
        return df

    def prepare_test_features(self, test_data, expected_features, target_column):
        
        
        logger.info(f"Preparing test features...")
        logger.info(f"Initial test data shape: {test_data.shape}")
        
        
        test_data_with_lags = self.create_lag_features(test_data, target_column)
        logger.info(f"After lag features: {test_data_with_lags.shape}")
        
       
        test_data_engineered = self.create_seasonal_features(test_data_with_lags)
        logger.info(f"After seasonal features: {test_data_engineered.shape}")
        
        
        columns_to_remove = ['date', 'city', 'AQI_Category']
        
        available_cols = [col for col in test_data_engineered.columns 
                         if col not in columns_to_remove and col != target_column]
        
        X_test_temp = test_data_engineered[available_cols].copy()
        y_test = test_data_engineered[target_column].copy()
        
        logger.info(f"Available feature columns: {len(available_cols)}")

        
        
        missing_features = []
        for feature in expected_features:
            if feature not in X_test_temp.columns:
                missing_features.append(feature)
              
                if 'lag' in str(feature) or 'rolling' in str(feature):
                  
                    X_test_temp[feature] = y_test.mean()
                else:
                    
                    X_test_temp[feature] = 0
        
        if missing_features:
            logger.info(f"Added missing features with defaults: {missing_features}")
        
        
        expected_features_clean = [str(f) for f in expected_features]
        
      
        for feature in expected_features_clean:
            if feature not in X_test_temp.columns:
                logger.info(f"Warning: Expected feature '{feature}' still missing, adding with default value")
                X_test_temp[feature] = 0
        
        X_test = X_test_temp[expected_features_clean].copy()
        
        logger.info(f"Final feature preparation:")
        logger.info(f"  X_test shape: {X_test.shape}")
        logger.info(f"  y_test shape: {y_test.shape}")
        logger.info(f"  Features match expected: {len(X_test.columns) == len(expected_features)}")
        logger.info(f"Final test features prepared with shape: {X_test.shape}")
        
        return X_test, y_test

    def evaluate(self):
        logger.info("Starting model evaluation...")
        
     
        model_artifacts = joblib.load(self.config.model_path)
        model = model_artifacts['model']
        target_column = model_artifacts['target_column']
        
        logger.info(f"Loaded model artifacts:")
        logger.info(f"  Model type: {model_artifacts.get('model_type', 'Unknown')}")
        logger.info(f"  Target column: {target_column}")
        
        
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            expected_features = model.feature_names_in_
            logger.info(f"Using model's feature_names_in_: {len(expected_features)} features")
        else:
            expected_features = model_artifacts['feature_columns']
            logger.info(f"Using saved feature_columns: {len(expected_features)} features")
        
      
        test_data = pd.read_csv(self.config.test_data_path, parse_dates=['date'])
        logger.info(f"Test data loaded: {test_data.shape}")
        
       
        X_test, y_test = self.prepare_test_features(test_data, expected_features, target_column)
        
        
        scaler = model_artifacts.get('scaler')
        if scaler is not None:
            logger.info("Checking scaler compatibility...")
            
           
            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = scaler.feature_names_in_
                logger.info(f"Scaler was fitted on {len(scaler_features)} features")
                
                
                current_features = set(X_test.columns)
                scaler_features_set = set(scaler_features)
                
                if current_features == scaler_features_set:
                    logger.info("Scaler features match perfectly, applying scaling...")
                    X_test_scaled = pd.DataFrame(
                        scaler.transform(X_test),
                        columns=X_test.columns,
                        index=X_test.index
                    )
                    X_test = X_test_scaled
                    logger.info("Scaling applied successfully")
                else:
                    logger.info("Scaler feature mismatch detected:")
                    logger.info(f"  Missing in current: {scaler_features_set - current_features}")
                    logger.info(f"  Extra in current: {current_features - scaler_features_set}")
                    logger.info("Skipping scaling to avoid errors...")
            else:
                logger.info("Scaler doesn't have feature_names_in_, attempting to apply scaling...")
                try:
                    X_test_scaled = pd.DataFrame(
                        scaler.transform(X_test),
                        columns=X_test.columns,
                        index=X_test.index
                    )
                    X_test = X_test_scaled
                    logger.info("Scaling applied successfully")
                except Exception as e:
                    logger.info(f"Scaling failed: {e}")
                    logger.info("Proceeding without scaling...")
        else:
            logger.info("No scaler found in model artifacts")
        
     
        logger.info("Making predictions...")
        y_pred = model.predict(X_test)
        logger.info(f"Predictions generated for {len(y_pred)} samples")

        self.predictions = y_pred
        self.actuals = y_test
        
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100
        
        results = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "MAPE": float(mape),
            "n_samples": int(len(y_test)),
            "target_column": target_column,
            "model_type": model_artifacts.get('model_type', 'Unknown')
        }
        
       
        os.makedirs(os.path.dirname(self.config.report_path), exist_ok=True)
        with open(self.config.report_path, "w") as f:
            json.dump(results, f, indent=4)
        
       
       
        logger.info(f"MAE (Mean Absolute Error): {mae:.4f}")
        logger.info(f"RMSE (Root Mean Square Error): {rmse:.4f}")
        logger.info(f"R² (R-squared): {r2:.4f}")
        logger.info(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        logger.info(f"Number of test samples: {len(y_test)}")
        
       
        
        
        if r2 < 0.3:
            logger.info("LOW R²: Model explains <30% of variance")
            logger.info("   Consider: More features, different algorithm, or data quality issues")
        elif r2 < 0.7:
            logger.info("MODERATE R²: Room for improvement")
        else:
            logger.info("GOOD R²: Model performs well")
            
        if mape > 20:
            logger.info("HIGH MAPE: >20% prediction error")
        elif mape > 10:
            logger.info(" MODERATE MAPE: 10-20% prediction error") 
        else:
            logger.info("LOW MAPE: <10% prediction error")
        
        
        pred_range = y_pred.max() - y_pred.min()
        actual_range = y_test.max() - y_test.min()
        range_coverage = pred_range / actual_range * 100
        
        logger.info(f"\nRange Coverage: {range_coverage:.1f}%")
        if range_coverage < 50:
            logger.info("Model predictions cover <50% of actual value range")
            logger.info("   This suggests the model may be underfitting")
        
        logger.info(f"\nResults saved to: {self.config.report_path}")
        
        
     
        logger.info(f"Prediction Range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
        logger.info(f"Actual Range: [{y_test.min():.2f}, {y_test.max():.2f}]")
        logger.info(f"Mean Absolute Residual: {np.mean(np.abs(y_test - y_pred)):.4f}")
        logger.info(f"Prediction Std: {y_pred.std():.4f}")
        logger.info(f"Actual Std: {y_test.std():.4f}")
        logger.info("Model Evaluation completed successfully!")
        
        return results
    
    def get_predictions(self):
        
        return self.predictions, self.actuals