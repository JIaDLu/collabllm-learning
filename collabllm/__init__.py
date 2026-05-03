"""
collabllm
~~~~~~~~~
Package initialisation: version metadata, global configuration flags,
and one-time setup (logging, runtime directories, …).
"""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Public package metadata                                                     #
# --------------------------------------------------------------------------- #
__version__ = "0.1.0"          # update as needed
__author__  = "Shirley Wu & the CollabLLM team"

__all__ = [
    "__version__",
    "ENABLE_COLLABLLM_LOGGING",
    "RUN_USER_DIR",
]

# --------------------------------------------------------------------------- #
# Utility: boolean env-var parser                                             #
# --------------------------------------------------------------------------- #
def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))
    
def _env_flag(name: str, default: str = "1") -> bool:
    """
    Convert an environment variable to bool.

    Truthy strings : "1", "true", "yes", "on"   (case-insensitive)
    Falsy  strings : "0", "false", "no", "off"
    """
    try:
        return bool(strtobool(os.getenv(name, default)))
    except ValueError:
        # Invalid value; fall back to default.
        return bool(strtobool(default))


# --------------------------------------------------------------------------- #
# Global logging switch                                                       #
# --------------------------------------------------------------------------- #
ENABLE_COLLABLLM_LOGGING: bool = _env_flag("ENABLE_COLLABLLM_LOGGING", "1")


_pkg_logger = logging.getLogger("collabllm")

if ENABLE_COLLABLLM_LOGGING:
    # Configure basic console output if the user hasn’t configured logging yet.
    # We guard with "if not root.handlers" to avoid double-configuration.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    _pkg_logger.info("CollabLLM logging enabled.")
else:
    # Silence *all* log records emitted from collabllm.* by:
    # 1) setting a level higher than CRITICAL
    # 2) preventing propagation to the root logger
    # 3) attaching a NullHandler
    _pkg_logger.setLevel(logging.CRITICAL)
    _pkg_logger.propagate = False
    _pkg_logger.handlers.clear()
    _pkg_logger.addHandler(logging.NullHandler())



