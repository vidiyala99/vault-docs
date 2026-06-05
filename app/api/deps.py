"""FastAPI dependency seams.

Every external resource the routes touch goes through one of these, so
tests (and future backends) swap them via `app.dependency_overrides`
instead of patching internals.
"""

from app.config import get_settings
from app.db import get_db  # re-exported: routes import all deps from here
from app.storage.local import LocalStorage

__all__ = ["get_db", "get_storage", "get_queue"]


def get_storage() -> LocalStorage:
    return LocalStorage(root=get_settings().storage_dir)


def get_queue():
    from redis import Redis
    from rq import Queue

    return Queue("processing", connection=Redis.from_url(get_settings().redis_url))
