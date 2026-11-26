"""The attachment blob files: one blob store per placement, beside that placement's hosted-room tables.

Blob bytes are ``hosted_room_blob_files`` rows of the placement's schema (file name to bytes), written in the verb's
transaction. The file steps: the verified read and the slice read with their size fence, the write, the unlink after
the commit and the orphan sweep.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from persistence.gateway.hosted_rooms import AttachmentIntegrityError, HostedPlacement

from .hosted_room_attachment_rules import BLOB_ID_RE, MAX_ATTACHMENT_BYTES
from ._hosted_store import BlobFiles, blob_files
from .hosted_room_tables import read_tables

__all__ = ["BlobFiles", "blob_files", "read_blob_slice", "read_verified_blob", "sweep_orphan_blobs", "unlink_blobs", "write_blob_file"]

# A blob's file name: its id, which must be a valid blob id.
def _blob_name(blob_id: str) -> str:
    if BLOB_ID_RE.fullmatch(blob_id) is None:
        raise AttachmentIntegrityError("stored blob id is invalid")
    return blob_id


def write_blob_file(files: BlobFiles, blob_id: str, data: bytes) -> None:
    """File step ``write_blob_file``: the atomic write lands the whole file under its blob name."""
    files[_blob_name(blob_id)] = bytes(data)


# A blob file's bytes, refused when the file is missing or its size changed.
def _open_blob(files: BlobFiles, blob_id: str, size: int) -> bytes:
    name = _blob_name(blob_id)
    if files.before_read is not None:
        files.before_read()
    data = files.get(name)
    if data is None:
        raise AttachmentIntegrityError("canonical attachment blob is unavailable")
    if len(data) != size:
        raise AttachmentIntegrityError("canonical attachment blob size changed")
    return data


def read_verified_blob(files: BlobFiles, *, blob_id: str, size: int, sha256: str) -> bytes:
    """File step ``read_verified_blob``: size fence, then the SHA-256 check."""
    data = _open_blob(files, blob_id, size)[:MAX_ATTACHMENT_BYTES + 1]
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise AttachmentIntegrityError("canonical attachment blob failed SHA-256 validation")
    return data


def read_blob_slice(files: BlobFiles, *, blob_id: str, size: int, offset: int, length: int) -> bytes:
    """File step ``read_blob_slice``: size fence, no hash."""
    data = _open_blob(files, blob_id, size)[offset:offset + length]
    if len(data) != min(length, size - offset):
        raise AttachmentIntegrityError("canonical attachment blob size changed")
    return data


def unlink_blobs(files: BlobFiles, blob_ids: list[str]) -> None:
    """The unlinks after a prune commits; a missing file is fine."""
    for blob_id in blob_ids:
        files.pop(_blob_name(blob_id), None)


def sweep_orphan_blobs(store: Any, placement: HostedPlacement) -> None:
    """Staging (``.tmp-``), quarantine (``.prune-``) and unreferenced ``blob_*`` files are reclaimed against the committed blob rows."""
    files = blob_files(store, placement)
    live = set(read_tables(store, placement)["hosted_room_attachment_blobs"])
    for name in list(files):
        if name.startswith((".tmp-", ".prune-")) or (BLOB_ID_RE.fullmatch(name) is not None and name not in live):
            del files[name]
