# utils/api_utils.py
import requests
import time
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, calls: int = 20, period: int = 60):
        self.calls = calls
        self.period = period
        self.timestamps = []

    def wait(self):
        now = time.time()
        self.timestamps = [ts for ts in self.timestamps if now - ts < self.period]
        
        if len(self.timestamps) >= self.calls:
            sleep_time = self.timestamps[0] + self.period - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.timestamps = self.timestamps[1:]
        
        self.timestamps.append(now)

class NBAAPIClient:
    def __init__(self, config):
        self.config = config
        self.rate_limiter = RateLimiter()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        })

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        self.rate_limiter.wait()
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {url} - {str(e)}")
            raise