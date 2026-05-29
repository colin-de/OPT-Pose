# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

import hashlib


def stable_seq_id(key: str, bits: int = 31) -> int:
    """Deterministically map a string key to a non-negative integer id."""
    key_bytes = str(key).encode("utf-8", errors="ignore")
    digest = hashlib.sha1(key_bytes).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value & ((1 << bits) - 1)
