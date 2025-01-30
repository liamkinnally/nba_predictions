# pipeline/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional
import pytz
from functools import wraps
import time

class PipelineScheduler:
    def __init__(self, daily_pipeline, historical_pipeline, timezone: str = 'US/Eastern'):
        self.daily_pipeline = daily_pipeline
        self.historical_pipeline = historical_pipeline
        self.timezone = pytz.timezone(timezone)
        
        # Initialize scheduler
        self.scheduler = BackgroundScheduler(
            timezone=self.timezone,
            job_defaults={'coalesce': True, 'max_instances': 1}
        )
        
        # Set up logging
        self.logger = logging.getLogger('PipelineScheduler')
        
        # Track job status
        self.job_status = {}
        
        # Set up error handling
        self.scheduler.add_listener(
            self._job_listener,
            EVENT_JOB_ERROR | EVENT_JOB_EXECUTED
        )

    def start(self):
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            self.logger.info("Scheduler started")
            
            # Schedule default jobs
            self._schedule_default_jobs()

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            self.logger.info("Scheduler stopped")

    def _retry_decorator(max_attempts: int = 3, delay: int = 300):
        """Decorator to handle job retries."""
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                attempts = 0
                while attempts < max_attempts:
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as e:
                        attempts += 1
                        self.logger.error(
                            f"Attempt {attempts}/{max_attempts} failed: {str(e)}"
                        )
                        if attempts < max_attempts:
                            time.sleep(delay)
                        else:
                            raise
            return wrapper
        return decorator

    @_retry_decorator(max_attempts=3, delay=300)
    def run_daily_predictions(self):
        """Execute daily predictions pipeline with retries."""
        self.logger.info("Starting daily predictions run")
        return self.daily_pipeline.run_pipeline()

    @_retry_decorator(max_attempts=2, delay=3600)
    def run_model_training(self):
        """Execute model training pipeline with retries."""
        self.logger.info("Starting model training run")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Train on last year of data
        return self.historical_pipeline.process_historical_data(start_date, end_date)

    def _schedule_default_jobs(self):
        """Schedule default pipeline jobs."""
        # Schedule daily predictions to run at 11:00 AM ET (before games typically start)
        self.scheduler.add_job(
            self.run_daily_predictions,
            CronTrigger(hour=11, minute=0, timezone=self.timezone),
            id='daily_predictions',
            name='Daily Predictions Pipeline'
        )
        
        # Schedule model training to run weekly on Monday at 4:00 AM ET
        self.scheduler.add_job(
            self.run_model_training,
            CronTrigger(day_of_week='mon', hour=4, minute=0, timezone=self.timezone),
            id='model_training',
            name='Weekly Model Training'
        )
        
        # Schedule odds updates every 15 minutes during active hours (11 AM - 11 PM ET)
        self.scheduler.add_job(
            self._update_odds,
            CronTrigger(
                hour='11-23',
                minute='*/15',
                timezone=self.timezone
            ),
            id='odds_updates',
            name='Odds Updates'
        )

    def add_job(
        self,
        func: callable,
        trigger: str,
        trigger_args: Dict[str, Any],
        job_id: str,
        name: str,
        **kwargs
    ):
        """Add a new job to the scheduler."""
        if trigger.lower() == 'cron':
            trigger = CronTrigger(**trigger_args, timezone=self.timezone)
        elif trigger.lower() == 'interval':
            trigger = IntervalTrigger(**trigger_args, timezone=self.timezone)
        else:
            raise ValueError(f"Unsupported trigger type: {trigger}")
            
        self.scheduler.add_job(
            func,
            trigger,
            id=job_id,
            name=name,
            **kwargs
        )
        self.logger.info(f"Added job: {name} ({job_id})")

    def remove_job(self, job_id: str):
        """Remove a job from the scheduler."""
        self.scheduler.remove_job(job_id)
        self.logger.info(f"Removed job: {job_id}")

    def get_job_status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of jobs."""
        if job_id:
            return self.job_status.get(job_id, {})
        return self.job_status

    def _job_listener(self, event):
        """Handle job execution events."""
        job_id = event.job_id
        
        if event.code == EVENT_JOB_EXECUTED:
            status = {
                'last_run': datetime.now(self.timezone),
                'status': 'success',
                'next_run': self.scheduler.get_job(job_id).next_run_time,
                'error': None
            }
        else:  # EVENT_JOB_ERROR
            status = {
                'last_run': datetime.now(self.timezone),
                'status': 'error',
                'next_run': self.scheduler.get_job(job_id).next_run_time,
                'error': str(event.exception)
            }
            
        self.job_status[job_id] = status
        
        if status['status'] == 'error':
            self.logger.error(f"Job {job_id} failed: {status['error']}")

    @_retry_decorator(max_attempts=2, delay=60)
    def _update_odds(self):
        """Update betting odds data."""
        self.logger.info("Updating odds data")
        odds = self.daily_pipeline.odds_collector.get_todays_odds()
        
        # Store updated odds in database
        with self.daily_pipeline.db_manager.get_session() as session:
            for game_id, game_odds in odds.items():
                game = session.query(Game).filter(Game.game_id == game_id).first()
                if game:
                    game.vegas_total = game_odds.get('vegas_total')
                    game.vegas_spread = game_odds.get('vegas_spread')
            session.commit()

# Example usage:
def setup_pipeline_scheduler():
    """Set up and start the pipeline scheduler."""
    # Initialize components
    db_manager = DatabaseManager()
    api_client = NBAAPIClient(Config())
    
    game_collector = GameCollector(api_client, db_manager)
    player_collector = PlayerCollector(api_client, db_manager)
    odds_collector = OddsCollector(api_client, db_manager)
    
    stats_processor = StatsProcessor(db_manager)
    feature_engineer = FeatureEngineer(db_manager, stats_processor)
    model_trainer = ModelTrainer(db_manager, feature_engineer, Config().MODEL_PATH)
    
    # Initialize pipelines
    daily_pipeline = DailyPipeline(
        db_manager,
        game_collector,
        player_collector,
        odds_collector,
        feature_engineer,
        model_trainer
    )
    
    historical_pipeline = HistoricalPipeline(
        db_manager,
        game_collector,
        player_collector,
        feature_engineer,
        model_trainer
    )
    
    # Initialize and start scheduler
    scheduler = PipelineScheduler(daily_pipeline, historical_pipeline)
    scheduler.start()
    
    return scheduler

# Example of adding a custom job:
if __name__ == "__main__":
    scheduler = setup_pipeline_scheduler()
    
    # Add custom job to update player stats every 6 hours
    scheduler.add_job(
        func=player_collector.update_player_stats,
        trigger='interval',
        trigger_args={'hours': 6},
        job_id='player_stats_update',
        name='Player Stats Update'
    )
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.stop()