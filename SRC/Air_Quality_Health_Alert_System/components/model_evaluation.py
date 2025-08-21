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

    def create_lag_features(self, df, target_col='aqi', lags=[1, 2, 3], rolling_windows=[3, 7]):
       
        df = df.copy()
        
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Creating lag features for {target_col}...")
        
       
        for lag in lags:
            df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
            
       
        for window in rolling_windows:
            df[f"{target_col}_rolling{window}"] = df[target_col].shift(1).rolling(window=window).mean()
            df[f"{target_col}_rolling{window}_std"] = df[target_col].shift(1).rolling(window=window).std()
        
 
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        print(f"Lag features created. Rows: {initial_rows} -> {final_rows} (removed {initial_rows - final_rows} NaN rows)")
        return df

    def create_seasonal_features(self, df):
        
        df = df.copy()
        
        if 'season' in df.columns:
            print("Creating one-hot encoded seasonal features...")
           
            seasonal_dummies = pd.get_dummies(df['season'], prefix='season', drop_first=False)
            
           
            df = pd.concat([df, seasonal_dummies], axis=1)
            
            
            df = df.drop('season', axis=1)
            
            print(f"Seasonal features created: {[col for col in seasonal_dummies.columns]}")
        else:
            print("Warning: 'season' column not found!")
        
        return df

    def prepare_test_features(self, test_data, expected_features, target_column):
        
        
        print(f"Preparing test features...")
        print(f"Initial test data shape: {test_data.shape}")
        
        
        test_data_with_lags = self.create_lag_features(test_data, target_column)
        print(f"After lag features: {test_data_with_lags.shape}")
        
       
        test_data_engineered = self.create_seasonal_features(test_data_with_lags)
        print(f"After seasonal features: {test_data_engineered.shape}")
        
        
        columns_to_remove = ['date', 'city', 'AQI_Category']
        
        available_cols = [col for col in test_data_engineered.columns 
                         if col not in columns_to_remove and col != target_column]
        
        X_test_temp = test_data_engineered[available_cols].copy()
        y_test = test_data_engineered[target_column].copy()
        
        print(f"Available feature columns: {len(available_cols)}")
        
        
        missing_features = []
        for feature in expected_features:
            if feature not in X_test_temp.columns:
                missing_features.append(feature)
              
                if 'lag' in str(feature) or 'rolling' in str(feature):
                  
                    X_test_temp[feature] = y_test.mean()
                else:
                    
                    X_test_temp[feature] = 0
        
        if missing_features:
            print(f"Added missing features with defaults: {missing_features}")
        
        
        expected_features_clean = [str(f) for f in expected_features]
        
      
        for feature in expected_features_clean:
            if feature not in X_test_temp.columns:
                print(f"Warning: Expected feature '{feature}' still missing, adding with default value")
                X_test_temp[feature] = 0
        
        X_test = X_test_temp[expected_features_clean].copy()
        
        print(f"Final feature preparation:")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")
        print(f"  Features match expected: {len(X_test.columns) == len(expected_features)}")
        
        return X_test, y_test

    def evaluate(self):
        print("Starting model evaluation...")
        
     
        model_artifacts = joblib.load(self.config.model_path)
        model = model_artifacts['model']
        target_column = model_artifacts['target_column']
        
        print(f"Loaded model artifacts:")
        print(f"  Model type: {model_artifacts.get('model_type', 'Unknown')}")
        print(f"  Target column: {target_column}")
        
        
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            expected_features = model.feature_names_in_
            print(f"Using model's feature_names_in_: {len(expected_features)} features")
        else:
            expected_features = model_artifacts['feature_columns']
            print(f"Using saved feature_columns: {len(expected_features)} features")
        
      
        test_data = pd.read_csv(self.config.test_data_path, parse_dates=['date'])
        print(f"Test data loaded: {test_data.shape}")
        
       
        X_test, y_test = self.prepare_test_features(test_data, expected_features, target_column)
        
        
        scaler = model_artifacts.get('scaler')
        if scaler is not None:
            print("Checking scaler compatibility...")
            
           
            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = scaler.feature_names_in_
                print(f"Scaler was fitted on {len(scaler_features)} features")
                
                
                current_features = set(X_test.columns)
                scaler_features_set = set(scaler_features)
                
                if current_features == scaler_features_set:
                    print("Scaler features match perfectly, applying scaling...")
                    X_test_scaled = pd.DataFrame(
                        scaler.transform(X_test),
                        columns=X_test.columns,
                        index=X_test.index
                    )
                    X_test = X_test_scaled
                    print("Scaling applied successfully")
                else:
                    print("Scaler feature mismatch detected:")
                    print(f"  Missing in current: {scaler_features_set - current_features}")
                    print(f"  Extra in current: {current_features - scaler_features_set}")
                    print("Skipping scaling to avoid errors...")
            else:
                print("Scaler doesn't have feature_names_in_, attempting to apply scaling...")
                try:
                    X_test_scaled = pd.DataFrame(
                        scaler.transform(X_test),
                        columns=X_test.columns,
                        index=X_test.index
                    )
                    X_test = X_test_scaled
                    print("Scaling applied successfully")
                except Exception as e:
                    print(f"Scaling failed: {e}")
                    print("Proceeding without scaling...")
        else:
            print("No scaler found in model artifacts")
        
     
        print("Making predictions...")
        y_pred = model.predict(X_test)
        print(f"Predictions generated for {len(y_pred)} samples")
        
        # Calculate metrics
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
        
       
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"MAE (Mean Absolute Error): {mae:.4f}")
        print(f"RMSE (Root Mean Square Error): {rmse:.4f}")
        print(f"R² (R-squared): {r2:.4f}")
        print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        print(f"Number of test samples: {len(y_test)}")
        
       
        print(f"\n" + "="*30)
        print("PERFORMANCE ANALYSIS")
        print("="*30)
        
        if r2 < 0.3:
            print("LOW R²: Model explains <30% of variance")
            print("   Consider: More features, different algorithm, or data quality issues")
        elif r2 < 0.7:
            print("MODERATE R²: Room for improvement")
        else:
            print("GOOD R²: Model performs well")
            
        if mape > 20:
            print("HIGH MAPE: >20% prediction error")
        elif mape > 10:
            print(" MODERATE MAPE: 10-20% prediction error") 
        else:
            print("LOW MAPE: <10% prediction error")
        
        
        pred_range = y_pred.max() - y_pred.min()
        actual_range = y_test.max() - y_test.min()
        range_coverage = pred_range / actual_range * 100
        
        print(f"\nRange Coverage: {range_coverage:.1f}%")
        if range_coverage < 50:
            print("Model predictions cover <50% of actual value range")
            print("   This suggests the model may be underfitting")
        
        print(f"\nResults saved to: {self.config.report_path}")
        
        
        print(f"\nDETAILED STATISTICS:")
        print(f"Prediction Range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
        print(f"Actual Range: [{y_test.min():.2f}, {y_test.max():.2f}]")
        print(f"Mean Absolute Residual: {np.mean(np.abs(y_test - y_pred)):.4f}")
        print(f"Prediction Std: {y_pred.std():.4f}")
        print(f"Actual Std: {y_test.std():.4f}")
        
        return results