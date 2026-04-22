"""
Shared utilities: logging configuration, directory helpers, label normalisation.
"""

import logging
import os

import config as cfg


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger with timestamped format.

    Args:
        level: Logging level (default ``logging.INFO``).

    Returns:
        The configured root logger.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(format=fmt, level=level, datefmt="%H:%M:%S")
    return logging.getLogger("fork_pipeline")


def ensure_output_dirs() -> None:
    """Create ``output/figures/`` if it does not already exist."""
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


def normalize_hand_label(raw: str) -> str:
    """Normalise a tremor-hand label to ``Right`` / ``Left`` / ``Bilateral``.

    Handles English values and common Hebrew equivalents.

    Args:
        raw: Raw string from the CRF cell.

    Returns:
        One of ``"Right"``, ``"Left"``, ``"Bilateral"``, or ``"Unknown"``.
    """
    if not isinstance(raw, str):
        return "Unknown"
    cleaned = raw.strip()
    mapping = {
        "right": "Right",
        "left": "Left",
        "bilateral": "Bilateral",
        "ימין": "Right",
        "שמאל": "Left",
    }
    return mapping.get(cleaned.lower(), "Unknown")
