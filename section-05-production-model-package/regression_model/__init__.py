import logging
import os
from pathlib import Path

from regression_model.config.core import PACKAGE_ROOT, config

# Configure null handler for library logging
logger = logging.getLogger(config.app_config.package_name)
logger.addHandler(logging.NullHandler())
logger.propagate = True


try:
    VERSION_PATH = PACKAGE_ROOT / "VERSION"
    if VERSION_PATH.is_file():
        with open(VERSION_PATH, "r", encoding="utf-8") as version_file:
            __version__ = version_file.read().strip()
    else:
        __version__ = "0.0.0"
except Exception as e:
    logger.warning(f"Failed to read VERSION file: {e}")
    __version__ = "0.0.0"


__all__ = ["__version__"]
