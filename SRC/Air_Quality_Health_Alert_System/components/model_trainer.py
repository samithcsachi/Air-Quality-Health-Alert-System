import pandas as pd
import os
import numpy as np
from datetime import datetime
from Air_Quality_Health_Alert_System import logger
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
from Air_Quality_Health_Alert_System.entity.config_entity  import ModelTrainerConfig
from pathlib import Path


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.feature_columns = None

    def load_scaler(self):
        
        scaler_path = os.path.join(self.config.data_transformation_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from: {scaler_path}")
        else:
            logger.warning(f"No scaler found at: {scaler_path}")
            self.scaler = None
   

    def prepare_features(self, train_data, test_data):
        
        
        columns_to_drop = ['date', 'city', 'AQI_Category']
        
        
        train_x = train_data.drop(columns=columns_to_drop + [self.config.target_column], errors='ignore')
        test_x = test_data.drop(columns=columns_to_drop + [self.config.target_column], errors='ignore')
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        
        cat_cols = train_x.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            logger.info(f"Encoding categorical columns: {cat_cols}")
            train_x = pd.get_dummies(train_x, columns=cat_cols, drop_first=True)
            test_x = pd.get_dummies(test_x, columns=cat_cols, drop_first=True)
            
            
            test_x = test_x.reindex(columns=train_x.columns, fill_value=0)

        
        self.feature_columns = train_x.columns.tolist()
        
        logger.info(f"Feature preparation completed:")
        logger.info(f"  Training features shape: {train_x.shape}")
        logger.info(f"  Test features shape: {test_x.shape}")
        logger.info(f"  Total features: {len(self.feature_columns)}")

        return train_x, test_x, train_y, test_y


    def get_hyperparameter_grid(self):
        
        param_grid = {
            'n_estimators': [100, 200, 300],  
            'max_depth': [3, 5, 7], 
            'learning_rate': [0.05, 0.1, 0.2], 
            'subsample': [0.7, 0.8, 0.9],  
            'colsample_bytree': [0.7, 0.8, 0.9],  
            'reg_alpha': [0, 0.1, 1],  
            'reg_lambda': [0, 0.1, 1],  
            'min_child_weight': [1, 3, 5],  
            'gamma': [0, 0.1, 0.2]  
        }
        return param_grid

    def evaluate_model(self, model, train_x, train_y, test_x, test_y):
       
        
      
        train_predictions = model.predict(train_x)
        test_predictions = model.predict(test_x)
        
       
        train_mse = mean_squared_error(train_y, train_predictions)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(train_y, train_predictions)
        train_r2 = r2_score(train_y, train_predictions)
        train_mape = np.mean(np.abs((train_y - train_predictions) / train_y)) * 100
        
        
        test_mse = mean_squared_error(test_y, test_predictions)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(test_y, test_predictions)
        test_r2 = r2_score(test_y, test_predictions)
        test_mape = np.mean(np.abs((test_y - test_predictions) / test_y)) * 100
        
        metrics = {
            'train': {
                'mse': train_mse,
                'rmse': train_rmse,
                'mae': train_mae,
                'r2': train_r2,
                'mape': train_mape
            },
            'test': {
                'mse': test_mse,
                'rmse': test_rmse,
                'mae': test_mae,
                'r2': test_r2,
                'mape': test_mape
            }
        }
        
        
        logger.info("=== Model Evaluation Metrics ===")
        logger.info("TRAINING SET:")
        logger.info(f"  MSE: {train_mse:.4f}")
        logger.info(f"  RMSE: {train_rmse:.4f}")
        logger.info(f"  MAE: {train_mae:.4f}")
        logger.info(f"  R²: {train_r2:.4f}")
        logger.info(f"  MAPE: {train_mape:.2f}%")
        
        logger.info("TEST SET:")
        logger.info(f"  MSE: {test_mse:.4f}")
        logger.info(f"  RMSE: {test_rmse:.4f}")
        logger.info(f"  MAE: {test_mae:.4f}")
        logger.info(f"  R²: {test_r2:.4f}")
        logger.info(f"  MAPE: {test_mape:.2f}%")
        
      
        r2_diff = train_r2 - test_r2
        if r2_diff > 0.1:
            logger.warning(f"Potential overfitting detected! Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        else:
            logger.info(f"Good generalization! Train-Test R² difference: {r2_diff:.4f}")
        
        return metrics, test_predictions

    def save_model_artifacts(self, model, metrics=None, test_data=None):
        os.makedirs(self.config.root_dir, exist_ok=True)
        
        model_artifacts = {
            'model': model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.config.target_column,
            'model_type': 'XGBRegressor',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        
        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(model_artifacts, model_path)
        logger.info(f"Model artifacts saved at: {model_path}")

        if test_data is not None:
            test_data_path = os.path.join(self.config.root_dir, "test_data.joblib")
            joblib.dump(test_data, test_data_path)
            logger.info(f"Test data saved at: {test_data_path}")
            
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_path = os.path.join(self.config.root_dir, "feature_importance.csv")
            feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved at: {importance_path}")
            
            
            logger.info("=== Top 10 Important Features ===")
            for idx, row in feature_importance.head(10).iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.4f}")

    def train(self):
       
        logger.info("Starting model training pipeline...")
        
        try:
            
            self.load_scaler()
            
            
            logger.info("Loading training and test data...")
            train_data = pd.read_csv(self.config.train_data_path, parse_dates=['date'])
            test_data = pd.read_csv(self.config.test_data_path, parse_dates=['date'])
            
            
            

            
            train_x, test_x, train_y, test_y = self.prepare_features(train_data, test_data)

          
            xgb_model = XGBRegressor(
                tree_method="hist", 
                random_state=self.config.random_state,
                n_jobs=-1
            )

            
            logger.info("Starting hyperparameter tuning...")
            param_grid = self.get_hyperparameter_grid()
            
           
            tscv = TimeSeriesSplit(n_splits=5)

            
            random_search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_grid,
                n_iter=30,  
                scoring='neg_mean_squared_error',
                cv=tscv,
                verbose=1,
                random_state=self.config.random_state,
                n_jobs=-1
            )

            
            logger.info("Training model with hyperparameter optimization...")
            random_search.fit(
                train_x, 
                train_y,
                eval_set=[(test_x, test_y)],
                early_stopping_rounds=10, 
                verbose=False
            )
            
            best_model = random_search.best_estimator_
            
            logger.info("=== Best Hyperparameters ===")
            for param, value in random_search.best_params_.items():
                logger.info(f"{param}: {value}")

            
            metrics, predictions = self.evaluate_model(best_model, train_x, train_y, test_x, test_y)

            test_data_to_save = {
            'X_test': test_x,
            'y_test': test_y,
            'predictions': predictions
        }

            
            self.save_model_artifacts(best_model, metrics, test_data_to_save)

            logger.info("Model training completed successfully!")
            
            return {
                'model': best_model,
                'metrics': metrics,
                'predictions': predictions,
                'feature_columns': self.feature_columns
            }
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise e
