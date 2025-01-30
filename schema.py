# database/schema.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Game(Base):
    __tablename__ = 'games'
    
    game_id = Column(String, primary_key=True)
    date = Column(DateTime, nullable=False)
    home_team_id = Column(String, ForeignKey('teams.team_id'), nullable=False)
    away_team_id = Column(String, ForeignKey('teams.team_id'), nullable=False)
    game_status = Column(String, nullable=False)
    vegas_total = Column(Float)
    vegas_spread = Column(Float)
    actual_total = Column(Float)
    
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    player_stats = relationship("PlayerGameStats", back_populates="game")

class Team(Base):
    __tablename__ = 'teams'
    
    team_id = Column(String, primary_key=True)
    team_name = Column(String, nullable=False)
    team_abbreviation = Column(String(3), nullable=False)
    conference = Column(String)
    division = Column(String)
    
    players = relationship("Player", back_populates="team")

class Player(Base):
    __tablename__ = 'players'
    
    player_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    team_id = Column(String, ForeignKey('teams.team_id'))
    position = Column(String)
    status = Column(String)
    
    team = relationship("Team", back_populates="players")
    game_stats = relationship("PlayerGameStats", back_populates="player")
    availability = relationship("PlayerAvailability", back_populates="player")

class PlayerGameStats(Base):
    __tablename__ = 'player_game_stats'
    
    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, ForeignKey('games.game_id'), nullable=False)
    player_id = Column(String, ForeignKey('players.player_id'), nullable=False)
    minutes_played = Column(Integer)
    points = Column(Integer)
    assists = Column(Integer)
    rebounds = Column(Integer)
    usage_rate = Column(Float)
    field_goal_attempts = Column(Integer)
    field_goals_made = Column(Integer)
    three_point_attempts = Column(Integer)
    three_points_made = Column(Integer)
    free_throw_attempts = Column(Integer)
    free_throws_made = Column(Integer)
    turnovers = Column(Integer)
    
    game = relationship("Game", back_populates="player_stats")
    player = relationship("Player", back_populates="game_stats")

class DailyProjection(Base):
    __tablename__ = 'daily_projections'
    
    projection_id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, ForeignKey('games.game_id'), nullable=False)
    player_id = Column(String, ForeignKey('players.player_id'), nullable=False)
    projected_points = Column(Float)
    projected_assists = Column(Float)
    projected_rebounds = Column(Float)
    confidence_score = Column(Float)
    timestamp = Column(DateTime, nullable=False)

class PlayerAvailability(Base):
    __tablename__ = 'player_availability'
    
    availability_id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String, ForeignKey('players.player_id'), nullable=False)
    game_id = Column(String, ForeignKey('games.game_id'), nullable=False)
    status = Column(Enum('starting', 'bench', 'out', name='player_status'), nullable=False)
    injury_note = Column(String)
    
    player = relationship("Player", back_populates="availability")