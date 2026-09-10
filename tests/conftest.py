import pytest

from ara.core.config import Settings
from ara.db.database import Database
from ara.db import UnitOfWork


@pytest.fixture()
def settings(tmp_path):
    return Settings(
        _env_file=None,
        env="test",
        database_url=f"sqlite:///./{tmp_path}/test.db",
        dev_api_key="test-key",
        colpali_mode="mock",
    )


@pytest.fixture()
def uow(settings):
    db = Database(settings.database_url)
    db.connect()
    store = UnitOfWork(db)
    yield store
    db.close()
