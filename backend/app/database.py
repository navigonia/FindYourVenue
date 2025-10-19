import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://distgis:distgis@db:5432/distgis"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      
    pool_size=10,            
    max_overflow=20,         
    future=True              
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()
