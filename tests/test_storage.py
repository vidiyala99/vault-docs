"""Storage seam: content-addressed local filesystem backend.

Keys are content hashes, which gives byte-level dedupe for free: uploading
the same file twice yields the same key, and the upload handler can detect
the duplicate before any processing is queued.
"""

import pytest

from app.storage.local import LocalStorage


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(root=tmp_path / "blobs")


class TestRoundTrip:
    def test_save_then_load_returns_same_bytes(self, storage):
        data = b"%PDF-1.4 fake document bytes"
        key = storage.save(data)
        assert storage.load(key) == data

    def test_exists(self, storage):
        key = storage.save(b"hello")
        assert storage.exists(key)
        assert not storage.exists("0" * 64)


class TestContentAddressing:
    def test_same_bytes_yield_same_key(self, storage):
        assert storage.save(b"identical") == storage.save(b"identical")

    def test_different_bytes_yield_different_keys(self, storage):
        assert storage.save(b"one") != storage.save(b"two")

    def test_key_is_a_hex_digest(self, storage):
        key = storage.save(b"data")
        assert len(key) == 64
        assert set(key) <= set("0123456789abcdef")


class TestErrors:
    def test_load_missing_key_raises(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.load("f" * 64)

    def test_empty_payload_rejected(self, storage):
        with pytest.raises(ValueError):
            storage.save(b"")
