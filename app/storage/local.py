"""Content-addressed local filesystem storage.

The key IS the sha256 of the bytes, so identical uploads collapse to one
blob and dedupe is a key comparison. Blobs are sharded two levels deep
(``ab/cd/abcd…``) to keep directory listings sane at thousands of files.
"""

import hashlib
from pathlib import Path


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LocalStorage:
    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    def save(self, data: bytes) -> str:
        if not data:
            raise ValueError("refusing to store an empty payload")
        key = content_hash(data)
        path = self._path(key)
        if not path.exists():  # idempotent: same bytes, same blob
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return key

    def load(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def _path(self, key: str) -> Path:
        return self._root / key[:2] / key[2:4] / key
