# models/model_trainer.py
from typing import Dict, Any, List
from datetime import datetime
import logging

class ModelTrainer:
    def __init__(self, db_manager, feature_engineer, model_dir: str):
        self.db_manager = db_manager
        self.feature_engineer = feature_engineer
        self.model_dir = model_dir
        
        # Initialize models
        self.models = {
            'points': PointsModel(db_manager, feature_engineer, model_dir),
            'assists': AssistsModel(db_manager, feature_engineer, model_dir),
            'rebounds': ReboundsModel(db_manager, feature_engineer, model_dir)
        }
    
    def train_all_models(self, start_date: datetime, end_date: datetime) -> Dict[str, Dict[str, float]]:
        """Train all models and return their metrics."""
        metrics = {}
        
        for name, model in self.models.items():
            try:
                logging.info(f"Training {name} model...")
                X, y = model.prepare_training_data(start_date, end_date)
                model_metrics = model.train(X, y)
                metrics[name] = model_metrics
                logging.info(f"Completed training {name} model. Metrics: {model_metrics}")
            except Exception as e:
                logging.error(f"Error training {name} model: {str(e)}")
                metrics[name] = {'error': str(e)}
        
        return metrics
    
    def load_all_models(self):
        """Load the latest version of all models."""
        for model in self.models.values():
            try:
                model.load_latest_model()
            except Exception as e:
                logging.error(f"Error loading {model.target_col} model: {str(e)}")