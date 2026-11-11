"""Alembic environment; the URL comes from application settings."""

from alembic import context

from app.core.config import get_settings
from app.core.database import make_engine
from app.models import Base

target_metadata = Base.metadata


def _url() -> str:
    return context.config.attributes.get("url") or get_settings().resolved_database_url


def run_migrations_online() -> None:
    get_settings().data_dir.mkdir(parents=True, exist_ok=True)  # default SQLite file lives here
    engine = make_engine(_url())
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
