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