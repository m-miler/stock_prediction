
from sqlalchemy.engine import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


class Settings:
    SQLALCHEMY_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@model_db:5432/model_info"
    engine: Engine = create_engine(SQLALCHEMY_DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


settings = Settings()
