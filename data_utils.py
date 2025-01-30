# utils/data_utils.py
from typing import Dict, List, Any, Union, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def validate_numeric(func):
    """Decorator to validate numeric inputs."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, (int, float)):
                if np.isnan(arg) or np.isinf(arg):
                    raise ValueError(f"Invalid numeric value: {arg}")
        for value in kwargs.values():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    raise ValueError(f"Invalid numeric value: {value}")
        return func(*args, **kwargs)
    return wrapper

class DataCleaner:
    @staticmethod
    def clean_player_name(name: str) -> str:
        """Standardize player name format."""
        if not name:
            return ""
        # Remove Jr., Sr., III, etc.
        suffixes = ['Jr.', 'Sr.', 'III', 'II', 'IV']
        cleaned_name = name
        for suffix in suffixes:
            cleaned_name = cleaned_name.replace(suffix, '').strip()
        # Convert to title case and strip whitespace
        return cleaned_name.title().strip()

    @staticmethod
    def standardize_team_name(team_name: str) -> str:
        """Standardize team name format."""
        team_mappings = {
            'PHX': 'Phoenix Suns',
            'LAL': 'Los Angeles Lakers',
            'LAC': 'Los Angeles Clippers',
            # Add more mappings as needed
        }
        return team_mappings.get(team_name.upper(), team_name)

    @staticmethod
    def clean_minutes_played(minutes: str) -> float:
        """Convert minutes played from MM:SS format to decimal."""
        try:
            if ':' in str(minutes):
                parts = minutes.split(':')
                return float(parts[0]) + float(parts[1])/60
            return float(minutes)
        except (ValueError, AttributeError):
            return 0.0

class StatsCalculator:
    @staticmethod
    @validate_numeric
    def calculate_ts_percentage(
        points: float,
        fga: float,
        fta: float
    ) -> float:
        """Calculate True Shooting Percentage."""
        try:
            ts_denominator = 2 * (fga + 0.44 * fta)
            if ts_denominator == 0:
                return 0.0
            return (points / ts_denominator) * 100
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    @validate_numeric
    def calculate_usg_percentage(
        fga: float,
        fta: float,
        tov: float,
        min_played: float,
        team_fga: float,
        team_fta: float,
        team_tov: float,
        team_min: float
    ) -> float:
        """Calculate Usage Percentage."""
        try:
            player_poss = fga + (0.44 * fta) + tov
            team_poss = team_fga + (0.44 * team_fta) + team_tov
            
            if team_poss == 0 or min_played == 0:
                return 0.0
                
            usg = 100 * (player_poss * (team_min / 5)) / (min_played * team_poss)
            return min(max(usg, 0), 100)  # Clamp between 0 and 100
        except ZeroDivisionError:
            return 0.0

class DataValidator:
    @staticmethod
    def validate_game_data(game_data: Dict[str, Any]) -> bool:
        """Validate game data structure."""
        required_fields = ['game_id', 'date', 'home_team_id', 'away_team_id']
        return all(field in game_data for field in required_fields)

    @staticmethod
    def validate_player_data(player_data: Dict[str, Any]) -> bool:
        """Validate player data structure."""
        required_fields = ['player_id', 'name', 'team_id']
        return all(field in player_data for field in required_fields)

    @staticmethod
    def validate_stats_data(stats_data: Dict[str, Any]) -> bool:
        """Validate player statistics data."""
        required_fields = [
            'minutes_played',
            'points',
            'assists',
            'rebounds',
            'field_goal_attempts',
            'free_throw_attempts'
        ]
        return all(field in stats_data for field in required_fields)

class DataTransformer:
    @staticmethod
    def calculate_rolling_stats(
        data: pd.DataFrame,
        columns: List[str],
        window: int = 5
    ) -> pd.DataFrame:
        """Calculate rolling averages for specified columns."""
        if data.empty:
            return pd.DataFrame()
            
        result = data.copy()
        for col in columns:
            if col in data.columns:
                result[f'{col}_rolling_avg'] = data[col].rolling(window=window, min_periods=1).mean()
        return result

    @staticmethod
    def normalize_features(
        data: pd.DataFrame,
        columns: List[str],
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """Normalize specified columns using various methods."""
        if data.empty:
            return pd.DataFrame()
            
        result = data.copy()
        for col in columns:
            if col in data.columns:
                if method == 'zscore':
                    mean = data[col].mean()
                    std = data[col].std()
                    if std != 0:
                        result[f'{col}_normalized'] = (data[col] - mean) / std
                elif method == 'minmax':
                    min_val = data[col].min()
                    max_val = data[col].max()
                    if max_val > min_val:
                        result[f'{col}_normalized'] = (data[col] - min_val) / (max_val - min_val)
        return result

def format_game_id(date: datetime, home_team: str, away_team: str) -> str:
    """Generate standardized game ID."""
    date_str = date.strftime('%Y%m%d')
    return f"{date_str}_{home_team}_{away_team}"

def parse_game_id(game_id: str) -> Dict[str, Union[datetime, str]]:
    """Parse components from game ID."""
    try:
        date_str, home_team, away_team = game_id.split('_')
        return {
            'date': datetime.strptime(date_str, '%Y%m%d'),
            'home_team': home_team,
            'away_team': away_team
        }
    except ValueError:
        logger.error(f"Invalid game ID format: {game_id}")
        return {}

def get_season_year(date: datetime) -> int:
    """Determine NBA season year from date."""
    if date.month >= 10:  # NBA season starts in October
        return date.year + 1
    return date.year

def calculate_rest_days(
    current_game_date: datetime,
    previous_game_date: Optional[datetime]
) -> int:
    """Calculate days of rest between games."""
    if previous_game_date is None:
        return 3  # Default to 3 days if no previous game
    delta = current_game_date - previous_game_date
    return max(0, delta.days)

def is_back_to_back(
    current_game_date: datetime,
    previous_game_date: Optional[datetime]
) -> bool:
    """Check if game is part of a back-to-back."""
    if previous_game_date is None:
        return False
    return (current_game_date - previous_game_date).days <= 1