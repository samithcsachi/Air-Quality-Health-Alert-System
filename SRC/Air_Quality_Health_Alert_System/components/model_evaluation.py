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
from datetime import datetime



class ModelEvaluation:
    def __init__(self, config):
        self.config = config
        self.predictions = None   
        self.actuals = None 

    def load_model_and_artifacts(self):
        
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found: {self.config.model_path}")
        
        model_artifacts = joblib.load(self.config.model_path)
        
        logger.info("Loaded model artifacts:")
        logger.info(f"  Model type: {model_artifacts.get('model_type', 'Unknown')}")
        logger.info(f"  Target column: {model_artifacts.get('target_column', 'Unknown')}")
        logger.info(f"  Timestamp: {model_artifacts.get('timestamp', 'Unknown')}")
        logger.info(f"  Features: {len(model_artifacts.get('feature_columns', []))}")
        
        return model_artifacts

    def load_test_data(self):
      
        if not os.path.exists(self.config.test_data_path):
            raise FileNotFoundError(f"Test data not found: {self.config.test_data_path}")
        
    
        if str(self.config.test_data_path).endswith('.csv'):
            test_data = pd.read_csv(self.config.test_data_path)
            logger.info(f"Loaded test data from CSV: {test_data.shape}")
        elif str(self.config.test_data_path).endswith('.joblib'):
            test_data = joblib.load(self.config.test_data_path)
            logger.info(f"Loaded test data from joblib: {type(test_data)}")
            
            
            if isinstance(test_data, dict):
                if 'X_test' in test_data and 'y_test' in test_data:
                    return test_data['X_test'], test_data['y_test']
                else:
                    raise ValueError("Joblib file doesn't contain 'X_test' and 'y_test' keys")
            elif isinstance(test_data, pd.DataFrame):
                # If it's a DataFrame, we need to split features and target
                if self.config.target_column in test_data.columns:
                    X_test = test_data.drop(columns=[self.config.target_column])
                    y_test = test_data[self.config.target_column]
                    return X_test, y_test
                else:
                    raise ValueError(f"Target column '{self.config.target_column}' not found in test data")
        else:
            raise ValueError(f"Unsupported file format: {self.config.test_data_path}")
        
        # For CSV data, split features and target
        if self.config.target_column in test_data.columns:
            X_test = test_data.drop(columns=[self.config.target_column])
            y_test = test_data[self.config.target_column]
            logger.info(f"Split test data - X: {X_test.shape}, y: {y_test.shape}")
            return X_test, y_test
        else:
            raise ValueError(f"Target column '{self.config.target_column}' not found in test data")

    def validate_feature_compatibility(self, X_test, expected_features):
        
        current_features = set(X_test.columns)
        expected_features_set = set(expected_features)
        
        missing_features = expected_features_set - current_features
        extra_features = current_features - expected_features_set
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            
        if extra_features:
            logger.warning(f"Extra features: {extra_features}")
        
        if missing_features or extra_features:
            logger.info("Reordering test features to match model expectations...")
            X_test = X_test.reindex(columns=expected_features, fill_value=0)
        
        logger.info(f"Feature compatibility check completed. Final shape: {X_test.shape}")
        return X_test

    def calculate_comprehensive_metrics(self, y_true, y_pred):
        
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
       
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
        
        
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
       
        pred_range = y_pred.max() - y_pred.min()
        actual_range = y_true.max() - y_true.min()
        range_coverage = (pred_range / actual_range * 100) if actual_range > 0 else 0
        
        
        pred_stats = {
            'mean': float(np.mean(y_pred)),
            'std': float(np.std(y_pred)),
            'min': float(np.min(y_pred)),
            'max': float(np.max(y_pred))
        }
        
        actual_stats = {
            'mean': float(np.mean(y_true)),
            'std': float(np.std(y_true)),
            'min': float(np.min(y_true)),
            'max': float(np.max(y_true))
        }
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'mean_residual': float(mean_residual),
            'std_residual': float(std_residual),
            'range_coverage': float(range_coverage),
            'n_samples': int(len(y_true)),
            'prediction_stats': pred_stats,
            'actual_stats': actual_stats
        }
        
        return metrics

    def interpret_results(self, metrics):
     
        logger.info("=== MODEL EVALUATION RESULTS ===")
        
       
        logger.info(f"MAE (Mean Absolute Error): {metrics['mae']:.4f}")
        logger.info(f"RMSE (Root Mean Square Error): {metrics['rmse']:.4f}")
        logger.info(f"R² (R-squared): {metrics['r2']:.4f}")
        logger.info(f"MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
        logger.info(f"Number of test samples: {metrics['n_samples']}")
        
        
        r2 = metrics['r2']
        mape = metrics['mape']
        
        logger.info("\n=== PERFORMANCE INTERPRETATION ===")
        
        # R² interpretation
        if r2 < 0.3:
            logger.info("LOW R²: Model explains <30% of variance")
            logger.info("   Recommendation: Consider more features, different algorithm, or data quality review")
        elif r2 < 0.7:
            logger.info("MODERATE R²: Room for improvement")
            logger.info("   Recommendation: Feature engineering or hyperparameter tuning")
        else:
            logger.info("GOOD R²: Model performs well")
            
        # MAPE interpretation
        if mape > 20:
            logger.info("HIGH MAPE: >20% prediction error")
            logger.info("   Recommendation: Model needs significant improvement")
        elif mape > 10:
            logger.info("MODERATE MAPE: 10-20% prediction error")
            logger.info("   Recommendation: Acceptable for some use cases, consider refinement")
        else:
            logger.info("LOW MAPE: <10% prediction error")
            logger.info("   Assessment: Good prediction accuracy")
        
        
        range_coverage = metrics['range_coverage']
        logger.info(f"\nRange Coverage: {range_coverage:.1f}%")
        if range_coverage < 50:
            logger.info("Low range coverage (<50%) - Model may be underfitting")
        elif range_coverage > 120:
            logger.info("High range coverage (>120%) - Model may be overpredicting variance")
        else:
            logger.info("Good range coverage")
        
        
        mean_residual = metrics['mean_residual']
        if abs(mean_residual) > metrics['mae'] * 0.1:
            logger.info(f"Systematic bias detected (mean residual: {mean_residual:.4f})")
        else:
            logger.info("No significant systematic bias")
        
       
        logger.info(f"\n=== PREDICTION STATISTICS ===")
        logger.info(f"Prediction Range: [{metrics['prediction_stats']['min']:.2f}, {metrics['prediction_stats']['max']:.2f}]")
        logger.info(f"Actual Range: [{metrics['actual_stats']['min']:.2f}, {metrics['actual_stats']['max']:.2f}]")
        logger.info(f"Prediction Mean: {metrics['prediction_stats']['mean']:.2f}")
        logger.info(f"Actual Mean: {metrics['actual_stats']['mean']:.2f}")

    def save_detailed_results(self, metrics, model_info):
      
        os.makedirs(self.config.root_dir, exist_ok=True)
        
        
        complete_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_type': model_info.get('model_type', 'Unknown'),
                'target_column': model_info.get('target_column', 'Unknown'),
                'model_timestamp': model_info.get('timestamp', 'Unknown'),
                'feature_count': len(model_info.get('feature_columns', []))
            },
            'metrics': metrics,
            'evaluation_config': {
                'model_path': str(self.config.model_path),
                'test_data_path': str(self.config.test_data_path)
            }
        }
        
      
        results_path = os.path.join(self.config.root_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(complete_results, f, indent=4)
        
        logger.info(f"Detailed results saved to: {results_path}")
        
     
        if self.predictions is not None and self.actuals is not None:
            predictions_df = pd.DataFrame({
                'actual': self.actuals.values if hasattr(self.actuals, 'values') else self.actuals,
                'predicted': self.predictions,
                'residual': (self.actuals.values if hasattr(self.actuals, 'values') else self.actuals) - self.predictions,
                'abs_residual': np.abs((self.actuals.values if hasattr(self.actuals, 'values') else self.actuals) - self.predictions),
                'percentage_error': np.abs(((self.actuals.values if hasattr(self.actuals, 'values') else self.actuals) - self.predictions) / 
                                         np.where((self.actuals.values if hasattr(self.actuals, 'values') else self.actuals) == 0, 1e-8, 
                                                (self.actuals.values if hasattr(self.actuals, 'values') else self.actuals))) * 100
            })
            
            predictions_path = os.path.join(self.config.root_dir, "predictions_vs_actuals.csv")
            predictions_df.to_csv(predictions_path, index=False)
            logger.info(f"Predictions vs actuals saved to: {predictions_path}")
        
       
        legacy_results = {
            "MAE": metrics['mae'],
            "RMSE": metrics['rmse'],
            "R2": metrics['r2'],
            "MAPE": metrics['mape'],
            "n_samples": metrics['n_samples'],
            "target_column": model_info.get('target_column', 'Unknown'),
            "model_type": model_info.get('model_type', 'Unknown')
        }
        
       
        if hasattr(self.config, 'report_path') and self.config.report_path:
            os.makedirs(os.path.dirname(self.config.report_path), exist_ok=True)
            with open(self.config.report_path, "w") as f:
                json.dump(legacy_results, f, indent=4)
            logger.info(f"Legacy report saved to: {self.config.report_path}")

    def evaluate(self):
        
        logger.info("Starting model evaluation...")
        
        try:
           
            model_artifacts = self.load_model_and_artifacts()
            model = model_artifacts['model']
            
            
            X_test, y_test = self.load_test_data()
            
            
            expected_features = model_artifacts.get('feature_columns', [])
            if expected_features:
                X_test = self.validate_feature_compatibility(X_test, expected_features)
            
            
            logger.info("Making predictions...")
            y_pred = model.predict(X_test)
            logger.info(f"Predictions generated for {len(y_pred)} samples")

           
            self.predictions = y_pred
            self.actuals = y_test
            
            
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred)
            
        
            self.interpret_results(metrics)
            
            
            self.save_detailed_results(metrics, model_artifacts)
            
            logger.info("Model evaluation completed successfully!")
            
            return {
                'metrics': metrics,
                'model_info': model_artifacts,
                'evaluation_completed': True
            }
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise e
    
