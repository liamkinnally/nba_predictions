# pipeline/daily_pipeline.py
from typing import Dict, List, Any
from datetime import datetime
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class DailyPipeline:
    def __init__(
        self,
        db_manager,
        game_collector,
        player_collector,
        odds_collector,
        feature_engineer,
        model_trainer
    ):
        self.db_manager = db_manager
        self.game_collector = game_collector
        self.player_collector = player_collector
        self.odds_collector = odds_collector
        self.feature_engineer = feature_engineer
        self.model_trainer = model_trainer
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DailyPipeline')

    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the full daily prediction pipeline."""
        try:
            self.logger.info("Starting daily prediction pipeline")
            pipeline_start = time.time()
            
            # Step 1: Get today's games and odds
            games = self._get_todays_games()
            if not games:
                self.logger.info("No games scheduled for today")
                return {}
            
            # Step 2: Get player availability
            active_players = self._get_active_players()
            
            # Step 3: Generate predictions for each player
            predictions = self._generate_predictions(games, active_players)
            
            # Step 4: Store predictions in database
            self._store_predictions(predictions)
            
            pipeline_duration = time.time() - pipeline_start
            self.logger.info(f"Pipeline completed in {pipeline_duration:.2f} seconds")
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _get_todays_games(self) -> List[Dict[str, Any]]:
        """Fetch today's games and associated odds."""
        self.logger.info("Fetching today's games and odds")
        
        try:
            # Get games and odds in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                games_future = executor.submit(self.game_collector.get_todays_games)
                odds_future = executor.submit(self.odds_collector.get_todays_odds)
                
                games = games_future.result()
                odds = odds_future.result()
            
            # Merge odds into games data
            for game in games:
                game_id = game['game_id']
                if game_id in odds:
                    game.update(odds[game_id])
            
            return games
        
        except Exception as e:
            self.logger.error(f"Error fetching games: {str(e)}")
            raise

    def _get_active_players(self) -> List[Dict[str, Any]]:
        """Get today's active players and their status."""
        self.logger.info("Fetching active players")
        
        try:
            lineups = self.player_collector.get_daily_lineups(datetime.now())
            
            # Filter to only include active players
            active_players = [
                player for player in lineups
                if player['status'] in ['active', 'starting']
            ]
            
            self.logger.info(f"Found {len(active_players)} active players")
            return active_players
        
        except Exception as e:
            self.logger.error(f"Error fetching active players: {str(e)}")
            raise

    def _generate_predictions(
        self,
        games: List[Dict[str, Any]],
        active_players: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate predictions for all active players."""
        self.logger.info("Generating predictions for active players")
        
        predictions = {}
        
        try:
            # Load all models first
            self.model_trainer.load_all_models()
            
            # Generate features and predictions for each player
            for player in active_players:
                player_id = player['player_id']
                
                # Find player's game
                game = self._find_player_game(player, games)
                if not game:
                    continue
                
                # Generate features
                features = self.feature_engineer.generate_player_features(
                    player_id,
                    game['game_id']
                )
                
                if not features:
                    continue
                
                # Convert features to DataFrame
                features_df = pd.DataFrame([features])
                
                # Generate predictions for each stat
                player_predictions = {}
                confidence_scores = {}
                
                for stat, model in self.model_trainer.models.items():
                    # Scale features
                    features_scaled = model.scaler.transform(features_df)
                    
                    # Make prediction
                    pred = model.model.predict(features_scaled)[0]
                    
                    # Calculate confidence score (simplified version)
                    confidence = self._calculate_confidence_score(
                        model,
                        features_scaled,
                        pred
                    )
                    
                    player_predictions[f'projected_{stat}'] = round(pred, 1)
                    confidence_scores[f'{stat}_confidence'] = round(confidence, 3)
                
                predictions[player_id] = {
                    'player_id': player_id,
                    'game_id': game['game_id'],
                    'name': player['name'],
                    'team_id': player['team_id'],
                    'timestamp': datetime.now(),
                    **player_predictions,
                    **confidence_scores
                }
            
            self.logger.info(f"Generated predictions for {len(predictions)} players")
            return predictions
        
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            raise

    def _find_player_game(
        self,
        player: Dict[str, Any],
        games: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find the game a player is participating in."""
        for game in games:
            if (game['home_team_id'] == player['team_id'] or
                game['away_team_id'] == player['team_id']):
                return game
        return None

    def _calculate_confidence_score(
        self,
        model: Any,
        features: np.ndarray,
        prediction: float
    ) -> float:
        """
        Calculate a confidence score for the prediction.
        This is a simplified version - could be enhanced with more sophisticated methods.
        """
        try:
            # Get feature importances
            importances = model.model.feature_importances_
            
            # Calculate weighted feature deviation from training mean
            feature_deviation = np.average(
                np.abs(features - np.mean(features, axis=0)),
                weights=importances
            )
            
            # Convert to confidence score (0 to 1)
            confidence = 1 / (1 + feature_deviation)
            
            return confidence
            
        except Exception:
            # Return default confidence if calculation fails
            return 0.5

    def _store_predictions(self, predictions: Dict[str, Dict[str, Any]]):
        """Store predictions in the database."""
        self.logger.info("Storing predictions in database")
        
        try:
            with self.db_manager.get_session() as session:
                for pred in predictions.values():
                    projection = DailyProjection(
                        game_id=pred['game_id'],
                        player_id=pred['player_id'],
                        projected_points=pred['projected_points'],
                        projected_assists=pred['projected_assists'],
                        projected_rebounds=pred['projected_rebounds'],
                        confidence_score=min([
                            pred['points_confidence'],
                            pred['assists_confidence'],
                            pred['rebounds_confidence']
                        ]),
                        timestamp=pred['timestamp']
                    )
                    session.add(projection)
                
                session.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing predictions: {str(e)}")
            raise
