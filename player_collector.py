# data/collectors/player_collector.py
from datetime import datetime
from typing import List, Dict, Any
import logging

class PlayerCollector:
    def __init__(self, api_client: NBAAPIClient, db_manager):
        self.api_client = api_client
        self.db_manager = db_manager

    def get_daily_lineups(self, date: datetime) -> List[Dict[str, Any]]:
        """Fetch daily lineups for a specific date."""
        date_str = date.strftime('%Y%m%d')
        url = self.api_client.config.API.DAILY_LINEUPS_URL.format(date=date_str)
        try:
            response = self.api_client._make_request(url)
            return self._parse_lineups_response(response)
        except Exception as e:
            logging.error(f"Failed to fetch lineups for {date_str}: {str(e)}")
            return []

    def _parse_lineups_response(self, response: Dict) -> List[Dict[str, Any]]:
        """Parse the raw lineups response into structured data."""
        lineups = []
        try:
            for player in response.get('data', []):
                parsed_player = {
                    'player_id': player.get('PLAYER_ID'),
                    'name': player.get('PLAYER_NAME'),
                    'team_id': player.get('TEAM_ID'),
                    'status': player.get('STATUS'),
                    'position': player.get('POSITION')
                }
                lineups.append(parsed_player)
        except Exception as e:
            logging.error(f"Error parsing lineups response: {str(e)}")
        return lineups
