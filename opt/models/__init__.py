# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

__all__ = ["OPT", "VGGT"]

def __getattr__(name):
    if name in __all__:
        from .opt import OPT, VGGT
        return {"OPT": OPT, "VGGT": VGGT}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(__all__)
