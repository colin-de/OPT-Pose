# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

# This module MUST be imported before any OpenGL-related imports
# It forcibly prevents OpenGL_accelerate from loading

import os
import sys

# Set environment variables BEFORE any OpenGL imports
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYOPENGL_USE_ACCELERATE"] = "False"

# Block OpenGL_accelerate from being imported
import builtins
_original_import = builtins.__import__

def _blocked_import(name, *args, **kwargs):
    if 'OpenGL_accelerate' in name:
        raise ImportError(f"OpenGL_accelerate is blocked: {name}")
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _blocked_import

# Now import OpenGL and configure it
try:
    import OpenGL
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_LOGGING = False  
    OpenGL.ERROR_ON_COPY = True
    OpenGL.ARRAY_SIZE_CHECKING = False
    OpenGL.FORWARD_COMPATIBLE_ONLY = False
    OpenGL.SIZE_1_ARRAY_UNPACK = False
    
    # Force PyOpenGL to use pure Python implementation
    import OpenGL.GL
    # Restore normal import after OpenGL is loaded
    builtins.__import__ = _original_import
    
    print("OpenGL initialized without accelerate module")
except Exception as e:
    builtins.__import__ = _original_import
    print(f"Warning during OpenGL initialization: {e}")

