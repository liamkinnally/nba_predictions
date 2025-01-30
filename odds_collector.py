# data/collectors/odds_collector.py
from typing import Dict, Any
import logging

class OddsCollector:
    def __init__(self, api_client: NBAAPIClient, db_manager):
        self.api_client = api_client
        self.db_manager = db_manager

    def get_todays_odds(self) -> Dict[str, Any]:
        """Fetch today's betting odds."""
        url = self.api_client.config.API.ODDS_URL
        try:
            response = self.api_client._make_request(url)
            return self._parse_odds_response(response)
        except Exception as e:
            logging.error(f"Failed to fetch odds: {str(e)}")
            return {}

    def _parse_odds_response(self, response: Dict) -> Dict[str, Dict[str, Any]]:
        """Parse the raw odds response into structured data."""
        odds_by_game = {}
        try:
            for game in response.get('games', []):
                game_id = game.get('gameId')
                odds_by_game[game_id] = {
                    'vegas_total': game.get('overUnder'),
                    'vegas_spread': game.get('spread'),
                    'home_team_odds': game.get('homeTeamOdds'),
                    'away_team_odds': game.get('awayTeamOdds')
                }
        except Exception as e:
            logging.error(f"Error parsing odds response: {str(e)}")
        return odds_by_game