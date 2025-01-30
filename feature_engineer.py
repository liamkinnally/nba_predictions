# data/processors/feature_engineer.py
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class FeatureEngineer:
    def __init__(self, db_manager, stats_processor: StatsProcessor):
        self.db_manager = db_manager
        self.stats_processor = stats_processor

    def generate_player_features(self, player_id: str, game_id: str) -> Dict[str, Any]:
        """Generate all features for a player for a specific game."""
        try:
            features = {}
            
            # Get basic player info and current game context
            game_context = self._get_game_context(game_id)
            features.update(game_context)
            
            # Calculate rolling averages
            rolling_stats = self.stats_processor.calculate_rolling_averages(player_id)
            features.update(rolling_stats)
            
            # Add matchup features
            matchup_stats = self._calculate_matchup_features(player_id, game_context['opponent_team_id'])
            features.update(matchup_stats)
            
            # Add team pace and recent performance
            team_features = self._calculate_team_features(game_context['team_id'])
            features.update(team_features)
            
            # Add rest days and back-to-back indicators
            schedule_features = self._calculate_schedule_features(player_id, game_id)
            features.update(schedule_features)
            
            return features
        except Exception as e:
            logging.error(f"Error generating features for player {player_id}: {str(e)}")
            return {}

    def _get_game_context(self, game_id: str) -> Dict[str, Any]:
        """Get contextual information about the game."""
        with self.db_manager.get_session() as session:
            game = session.query(Game).filter(Game.game_id == game_id).first()
            
            if not game:
                return {}
                
            return {
                'team_id': game.home_team_id,  # Will be updated based on player's team
                'opponent_team_id': game.away_team_id,  # Will be updated based on player's team
                'is_home_game': True,  # Will be updated based on player's team
                'vegas_total': game.vegas_total,
                'vegas_spread': game.vegas_spread
            }

    def _calculate_matchup_features(self, player_id: str, opponent_team_id: str) -> Dict[str, float]:
        """Calculate features based on historical performance against the opponent."""
        with self.db_manager.get_session() as session:
            # Get last 5 games against this opponent
            recent_matchups = pd.read_sql(
                session.query(PlayerGameStats)
                .join(Game)
                .filter(
                    PlayerGameStats.player_id == player_id,
                    (Game.home_team_id == opponent_team_id) | (Game.away_team_id == opponent_team_id)
                )
                .order_by(Game.date.desc())
                .limit(5)
                .statement,
                session.bind
            )
            
            if recent_matchups.empty:
                return {
                    'vs_opponent_points_avg': 0.0,
                    'vs_opponent_assists_avg': 0.0,
                    'vs_opponent_rebounds_avg': 0.0
                }
            
            return {
                'vs_opponent_points_avg': round(recent_matchups['points'].mean(), 2),
                'vs_opponent_assists_avg': round(recent_matchups['assists'].mean(), 2),
                'vs_opponent_rebounds_avg': round(recent_matchups['rebounds'].mean(), 2)
            }

    def _calculate_team_features(self, team_id: str) -> Dict[str, float]:
        """Calculate team-level features including pace and recent performance."""
        with self.db_manager.get_session() as session:
            # Get last 10 games for the team
            recent_games = pd.read_sql(
                session.query(Game)
                .filter((Game.home_team_id == team_id) | (Game.away_team_id == team_id))
                .order_by(Game.date.desc())
                .limit(10)
                .statement,
                session.bind
            )
            
            if recent_games.empty:
                return {
                    'team_pace': 0.0,
                    'team_recent_win_pct': 0.0
                }
            
            # Calculate pace and winning percentage
            # This is a simplified pace calculation
            pace = recent_games['actual_total'].mean() / 2
            
            return {
                'team_pace': round(pace, 2),
                'team_recent_win_pct': 0.0  # Would need to add logic to calculate wins
            }

    def _calculate_schedule_features(self, player_id: str, game_id: str) -> Dict[str, Any]:
        """Calculate features related to schedule such as rest days and back-to-back games."""
        with self.db_manager.get_session() as session:
            game = session.query(Game).filter(Game.game_id == game_id).first()
            
            if not game:
                return {
                    'days_rest': 0,
                    'is_back_to_back': False
                }
            
            # Find previous game
            prev_game = session.query(Game)\
                .join(PlayerGameStats)\
                .filter(
                    PlayerGameStats.player_id == player_id,
                    Game.date < game.date
                )\
                .order_by(Game.date.desc())\
                .first()
            
            if not prev_game:
                return {
                    'days_rest': 3,  # Default to 3 days rest if no previous game found
                    'is_back_to_back': False
                }
            
            days_rest = (game.date - prev_game.date).days
            
            return {
                'days_rest': days_rest,
                'is_back_to_back': days_rest == 1
            }