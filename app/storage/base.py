"""Storage seam.

Anything that can save/load/test blobs by key satisfies this Protocol.
Local filesystem today; an S3 backend is one class implementing the same
three methods — no call-site changes.
"""

from typing import Protocol


class Storage(Protocol):
    def save(self, data: bytes) -> str:
        """Persist ``data``, returning its content-derived key."""
        ...

    def load(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...
