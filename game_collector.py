# data/collectors/game_collector.py
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

class GameCollector:
    def __init__(self, api_client: NBAAPIClient, db_manager):
        self.api_client = api_client
        self.db_manager = db_manager

    def get_todays_games(self) -> List[Dict[str, Any]]:
        """Fetch today's games from the NBA API."""
        url = self.api_client.config.API.TODAYS_GAMES_URL
        try:
            response = self.api_client._make_request(url)
            return self._parse_games_response(response)
        except Exception as e:
            logging.error(f"Failed to fetch today's games: {str(e)}")
            return []

    def get_boxscore(self, game_id: str) -> Dict[str, Any]:
        """Fetch boxscore for a specific game."""
        url = self.api_client.config.API.BOXSCORE_URL.format(game_id=game_id)
        try:
            return self.api_client._make_request(url)
        except Exception as e:
            logging.error(f"Failed to fetch boxscore for game {game_id}: {str(e)}")
            return {}

    def _parse_games_response(self, response: Dict) -> List[Dict[str, Any]]:
        """Parse the raw games response into structured data."""
        games = []
        try:
            for game in response.get('scoreboard', {}).get('games', []):
                parsed_game = {
                    'game_id': game.get('gameId'),
                    'date': game.get('gameTimeUTC'),
                    'home_team_id': game.get('homeTeam', {}).get('teamId'),
                    'away_team_id': game.get('awayTeam', {}).get('teamId'),
                    'game_status': game.get('gameStatus'),
                }
                games.append(parsed_game)
        except Exception as e:
            logging.error(f"Error parsing games response: {str(e)}")
        return games
