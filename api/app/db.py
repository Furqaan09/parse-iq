from sqlmodel import Session, create_engine

from app.settings import settings

engine = create_engine(settings.database_url, echo=False)


# Context manager to get a session for database operations
def get_session():
    with Session(engine) as session:
        yield session
