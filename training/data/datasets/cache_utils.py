# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

import errno
import hashlib
import os
import random
import tempfile
import time
from typing import Optional


def _should_retry(err: OSError) -> bool:
    return err.errno in {
        errno.EPERM,
        errno.EACCES,
        errno.EROFS,
        errno.ENOSYS,
        errno.ENOENT,  # No such file or directory - can happen in multi-process scenarios
    }


def resolve_cache_dir(preferred_dir: str, dataset_name: str, data_root: Optional[str] = None) -> str:
    """Return a writable cache directory with graceful fallbacks.

    Args:
        preferred_dir: First-choice cache location (typically under the dataset root).
        dataset_name: Name used to namespace fallback cache directories.
        data_root: Dataset root path, used to derive a stable fallback subdirectory.

    The function attempts the preferred directory first. On failure due to
    read-only or unsupported filesystems, it falls back to locations specified by
    the `NOCS3R_CACHE_ROOT` environment variable, the user's `~/.cache` folder,
    and finally the system temporary directory. The returned directory is
    guaranteed to exist.
    """

    unique_suffix = ""
    if data_root:
        digest = hashlib.md5(data_root.encode("utf-8")).hexdigest()
        unique_suffix = digest[:10]

    def _with_suffix(path: str) -> str:
        if not unique_suffix:
            return path
        return os.path.join(path, dataset_name, unique_suffix)

    candidates = []
    if preferred_dir:
        candidates.append(preferred_dir)

    env_root = os.environ.get("NOCS3R_CACHE_ROOT")
    if env_root:
        candidates.append(_with_suffix(env_root))

    home_cache = _with_suffix(os.path.join(os.path.expanduser("~").replace("dsshome1", "mcmlscratch"), "code", "nocs3r", "cache"))
    candidates.append(home_cache)

    tmp_cache = _with_suffix(os.path.join(tempfile.gettempdir(), "nocs3r"))
    candidates.append(tmp_cache)

    last_error = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            
            # Add a small retry loop for write test to handle race conditions in multi-process scenarios
            max_write_attempts = 3
            write_success = False
            
            for attempt in range(max_write_attempts):
                try:
                    # Test if directory is actually writable by creating a temporary file
                    # Use process ID to avoid conflicts between processes
                    test_file = os.path.join(candidate, f".write_test_{os.getpid()}_{attempt}")
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    write_success = True
                    break
                except OSError as write_exc:
                    last_error = write_exc
                    if attempt < max_write_attempts - 1:
                        # Wait a bit before retrying (with jitter to reduce thundering herd)
                        time.sleep(0.01 + random.uniform(0, 0.02))
                        # Re-create directory in case it was removed
                        try:
                            os.makedirs(candidate, exist_ok=True)
                        except:
                            pass
                    else:
                        # Last attempt failed
                        if not _should_retry(write_exc):
                            raise
            
            if not write_success:
                # All write attempts failed, try next candidate
                continue
            
            if os.path.abspath(candidate) != os.path.abspath(preferred_dir):
                print(f"Using cache directory fallback: {candidate}")
            return candidate
        except OSError as exc:
            last_error = exc
            if not _should_retry(exc):
                raise
    raise RuntimeError(
        f"Unable to create cache directory. Last attempted path: {candidate}, error: {last_error}"
    )


