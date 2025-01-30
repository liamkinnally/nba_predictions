# models/base_model.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime
import os

class BaseModel(ABC):
    def __init__(self, db_manager, feature_engineer, target_col: str, model_dir: str):
        self.db_manager = db_manager
        self.feature_engineer = feature_engineer
        self.target_col = target_col
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def prepare_training_data(self, start_date: datetime, end_date: datetime) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training."""
        with self.db_manager.get_session() as session:
            # Get all games and stats in date range
            query = """
                SELECT 
                    pgs.*, g.date, g.vegas_total, g.vegas_spread,
                    g.home_team_id, g.away_team_id
                FROM player_game_stats pgs
                JOIN games g ON pgs.game_id = g.game_id
                WHERE g.date BETWEEN :start_date AND :end_date
                AND pgs.minutes_played > 0
            """
            
            df = pd.read_sql(
                query,
                session.bind,
                params={'start_date': start_date, 'end_date': end_date}
            )
            
            if df.empty:
                raise ValueError("No data found for the specified date range")
            
            # Generate features for each player-game combination
            features_list = []
            for _, row in df.iterrows():
                features = self.feature_engineer.generate_player_features(
                    row['player_id'],
                    row['game_id']
                )
                features['game_id'] = row['game_id']
                features['player_id'] = row['player_id']
                features_list.append(features)
            
            features_df = pd.DataFrame(features_list)
            
            # Merge features with actual results
            training_data = pd.merge(
                features_df,
                df[[self.target_col, 'game_id', 'player_id']],
                on=['game_id', 'player_id']
            )
            
            # Drop non-feature columns
            X = training_data.drop([self.target_col, 'game_id', 'player_id'], axis=1)
            y = training_data[self.target_col]
            
            return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model and return metrics."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = self._train_model(X_train_scaled, y_train)
            
            # Calculate metrics
            metrics = self._calculate_metrics(X_test_scaled, y_test)
            
            # Save model and scaler
            self._save_model()
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise
    
    @abstractmethod
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Implement specific model training logic."""
        pass
    
    def _calculate_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics."""
        predictions = self.model.predict(X)
        
        metrics = {
            'mae': np.mean(np.abs(predictions - y)),
            'rmse': np.sqrt(np.mean((predictions - y) ** 2)),
            'r2': self.model.score(X, y)
        }
        
        return {k: round(v, 3) for k, v in metrics.items()}
    
    def _save_model(self):
        """Save model and scaler to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.model_dir, f'{self.target_col}_model_{timestamp}.joblib')
        scaler_path = os.path.join(self.model_dir, f'{self.target_col}_scaler_{timestamp}.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
    def load_latest_model(self):
        """Load the most recent model and scaler from disk."""
        model_files = [f for f in os.listdir(self.model_dir) 
                      if f.startswith(f'{self.target_col}_model_')]
        scaler_files = [f for f in os.listdir(self.model_dir) 
                       if f.startswith(f'{self.target_col}_scaler_')]
        
        if not model_files or not scaler_files:
            raise FileNotFoundError("No saved models found")
            
        latest_model = max(model_files)
        latest_scaler = max(scaler_files)
        
        self.model = joblib.load(os.path.join(self.model_dir, latest_model))
        self.scaler = joblib.load(os.path.join(self.model_dir, latest_scaler))

# models/points_model.py
from sklearn.ensemble import GradientBoostingRegressor
from typing import Any
import numpy as np

class PointsModel(BaseModel):
    def __init__(self, db_manager, feature_engineer, model_dir: str):
        super().__init__(db_manager, feature_engineer, 'points', model_dir)
    
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        return model.fit(X, y)

# models/assists_model.py
from sklearn.ensemble import GradientBoostingRegressor
from typing import Any
import numpy as np

class AssistsModel(BaseModel):
    def __init__(self, db_manager, feature_engineer, model_dir: str):
        super().__init__(db_manager, feature_engineer, 'assists', model_dir)
    
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        return model.fit(X, y)

# models/rebounds_model.py
from sklearn.ensemble import GradientBoostingRegressor
from typing import Any
import numpy as np

class ReboundsModel(BaseModel):
    def __init__(self, db_manager, feature_engineer, model_dir: str):
        super().__init__(db_manager, feature_engineer, 'rebounds', model_dir)
    
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        return model.fit(X, y)

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