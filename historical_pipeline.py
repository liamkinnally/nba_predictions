# pipeline/historical_pipeline.py
class HistoricalPipeline:
    def __init__(
        self,
        db_manager,
        game_collector,
        player_collector,
        feature_engineer,
        model_trainer
    ):
        self.db_manager = db_manager
        self.game_collector = game_collector
        self.player_collector = player_collector
        self.feature_engineer = feature_engineer
        self.model_trainer = model_trainer
        self.logger = logging.getLogger('HistoricalPipeline')

    def process_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Process historical data and train models."""
        try:
            self.logger.info(f"Processing historical data from {start_date} to {end_date}")
            
            # Train models on historical data
            metrics = self.model_trainer.train_all_models(start_date, end_date)
            
            self.logger.info("Historical processing complete")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Historical processing failed: {str(e)}")
            raise

# utils/pipeline_utils.py
def run_daily_update():
    """Utility function to run the daily pipeline."""
    try:
        # Initialize components
        db_manager = DatabaseManager()
        api_client = NBAAPIClient(Config())
        
        game_collector = GameCollector(api_client, db_manager)
        player_collector = PlayerCollector(api_client, db_manager)
        odds_collector = OddsCollector(api_client, db_manager)
        
        stats_processor = StatsProcessor(db_manager)
        feature_engineer = FeatureEngineer(db_manager, stats_processor)
        
        model_trainer = ModelTrainer(db_manager, feature_engineer, Config().MODEL_PATH)
        
        # Initialize and run pipeline
        pipeline = DailyPipeline(
            db_manager,
            game_collector,
            player_collector,
            odds_collector,
            feature_engineer,
            model_trainer
        )
        
        predictions = pipeline.run_pipeline()
        return predictions
        
    except Exception as e:
        logging.error(f"Daily update failed: {str(e)}")
        raise