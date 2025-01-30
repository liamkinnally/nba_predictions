# database/database_manager.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import Generator
import os

class DatabaseManager:
    def __init__(self, connection_string: str = None):
        if connection_string is None:
            connection_string = os.getenv('DATABASE_URL', 'sqlite:///nba_predictions.db')
        
        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(self.engine)
    
    @contextmanager
    def get_session(self) -> Generator:
        """Provide a transactional scope around a series of operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()