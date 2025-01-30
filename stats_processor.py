# data/processors/stats_processor.py
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class StatsProcessor:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def calculate_usage_rate(self, stats: Dict[str, Any], team_stats: Dict[str, Any]) -> float:
        """
        Calculate usage rate for a player using the formula:
        100 * ((FGA + 0.44 * FTA + TOV) * (Tm MP / 5)) / (MP * (Tm FGA + 0.44 * Tm FTA + Tm TOV))
        """
        try:
            player_possessions = (stats['field_goal_attempts'] + 
                                0.44 * stats['free_throw_attempts'] + 
                                stats['turnovers'])
            
            team_possessions = (team_stats['field_goal_attempts'] + 
                              0.44 * team_stats['free_throw_attempts'] + 
                              team_stats['turnovers'])
            
            minutes_factor = (team_stats['total_minutes'] / 5) / stats['minutes_played']
            
            usage_rate = 100 * (player_possessions * minutes_factor) / team_possessions
            return round(usage_rate, 2)
        except (KeyError, ZeroDivisionError) as e:
            logging.error(f"Error calculating usage rate: {str(e)}")
            return 0.0

    def calculate_rolling_averages(self, player_id: str, days: int = 30) -> Dict[str, float]:
        """Calculate rolling averages for key statistics over the specified number of days."""
        with self.db_manager.get_session() as session:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Fetch recent games for the player
            recent_stats = pd.read_sql(
                session.query(PlayerGameStats)
                .filter(PlayerGameStats.player_id == player_id)
                .filter(Game.date >= cutoff_date)
                .join(Game)
                .statement,
                session.bind
            )
            
            if recent_stats.empty:
                return {}
            
            averages = {
                'points_avg': recent_stats['points'].mean(),
                'assists_avg': recent_stats['assists'].mean(),
                'rebounds_avg': recent_stats['rebounds'].mean(),
                'minutes_avg': recent_stats['minutes_played'].mean(),
                'usage_rate_avg': recent_stats['usage_rate'].mean()
            }
            
            return {k: round(v, 2) for k, v in averages.items()}

    def process_game_stats(self, boxscore_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw boxscore data into structured game statistics."""
        processed_stats = {}
        
        try:
            game_data = boxscore_data.get('game', {})
            
            for team in ['homeTeam', 'awayTeam']:
                team_data = game_data.get(team, {})
                team_id = team_data.get('teamId')
                
                team_stats = self._process_team_stats(team_data)
                player_stats = self._process_player_stats(team_data.get('players', []), team_stats)
                
                processed_stats[team_id] = {
                    'team_stats': team_stats,
                    'player_stats': player_stats
                }
            
            return processed_stats
        except Exception as e:
            logging.error(f"Error processing game stats: {str(e)}")
            return {}

    def _process_team_stats(self, team_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and calculate team-level statistics."""
        stats = team_data.get('statistics', {})
        return {
            'field_goal_attempts': int(stats.get('fieldGoalsAttempted', 0)),
            'free_throw_attempts': int(stats.get('freeThrowsAttempted', 0)),
            'turnovers': int(stats.get('turnovers', 0)),
            'total_minutes': 240,  # Standard game length
            'points': int(stats.get('points', 0)),
            'assists': int(stats.get('assists', 0)),
            'rebounds': int(stats.get('reboundsTotal', 0))
        }

    def _process_player_stats(self, players_data: List[Dict], team_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process individual player statistics."""
        processed_players = []
        
        for player in players_data:
            stats = player.get('statistics', {})
            
            processed_stats = {
                'player_id': player.get('personId'),
                'minutes_played': int(stats.get('minutesCalculated', '0').split(':')[0]),
                'points': int(stats.get('points', 0)),
                'assists': int(stats.get('assists', 0)),
                'rebounds': int(stats.get('reboundsTotal', 0)),
                'field_goal_attempts': int(stats.get('fieldGoalsAttempted', 0)),
                'free_throw_attempts': int(stats.get('freeThrowsAttempted', 0)),
                'turnovers': int(stats.get('turnovers', 0))
            }
            
            # Calculate usage rate if player played minutes
            if processed_stats['minutes_played'] > 0:
                processed_stats['usage_rate'] = self.calculate_usage_rate(processed_stats, team_stats)
            else:
                processed_stats['usage_rate'] = 0.0
                
            processed_players.append(processed_stats)
            
        return processed_players