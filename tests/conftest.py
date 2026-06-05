"""Shared fixtures.

DB-backed tests run against a throwaway `vault_test` database on the real
pgvector Postgres from docker-compose. If Postgres isn't running, those
tests skip (the pure-unit suite stays green with zero infrastructure).
"""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

_ADMIN_URL = "postgresql+psycopg://vault:vault@localhost:5433/vault"
_TEST_URL = "postgresql+psycopg://vault:vault@localhost:5433/vault_test"


def _db_available() -> bool:
    try:
        engine = create_engine(_ADMIN_URL, connect_args={"connect_timeout": 2})
        with engine.connect():
            return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _db_available(), reason="Postgres not running (docker compose up -d)"
)


@pytest.fixture(scope="session")
def test_engine():
    admin = create_engine(_ADMIN_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text("DROP DATABASE IF EXISTS vault_test WITH (FORCE)"))
        conn.execute(text("CREATE DATABASE vault_test"))
    engine = create_engine(_TEST_URL)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    from app import models  # noqa: F401  (register tables)
    from app.db import Base

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session_factory(test_engine):
    """Fresh tables for every test: truncate everything between runs."""
    from app.db import Base

    with test_engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f'TRUNCATE TABLE "{table.name}" CASCADE'))
    return sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)
