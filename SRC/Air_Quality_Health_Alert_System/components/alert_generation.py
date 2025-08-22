import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from Air_Quality_Health_Alert_System import logger
from Air_Quality_Health_Alert_System.entity.config_entity import AlertConfig

class AlertGenerator:
    def __init__(self, config: AlertConfig):
        self.config = config

    def check_threshold(self, predicted_aqi):
        sensitive_alert = predicted_aqi > self.config.sensitive_threshold
        general_alert = predicted_aqi > self.config.general_threshold
        messages = []
        if sensitive_alert:
            messages.append("Alert: AQI above sensitive threshold!")
        if general_alert:
            messages.append("Alert: AQI above general threshold!")
        logger.info(f"Thresholds checked: Sensitive Alert={sensitive_alert}, General Alert={general_alert}")
        return sensitive_alert, general_alert, messages

    def get_alerts_for_dashboard(self, predicted_aqi):
        _, _, messages = self.check_threshold(predicted_aqi)
        return messages

    def evaluate_alerts(self, predicted_aqi, actual_aqi):
        if hasattr(predicted_aqi, 'values'):
            predicted_aqi = predicted_aqi.values
        if hasattr(actual_aqi, 'values'):
            actual_aqi = actual_aqi.values
            
        y_true = (actual_aqi > self.config.sensitive_threshold).astype(int)
        y_pred = (predicted_aqi > self.config.sensitive_threshold).astype(int)
        logger.info("Alert evaluation metrics calculated successfully!")
        
        return {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
