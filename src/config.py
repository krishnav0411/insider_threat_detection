# src/config.py
"""
Central configuration loader for the Insider Threat Detection system.
Reads config.yaml and exports typed constants used by every src module.

Usage:
    >>> from src.config import settings
    >>> settings["model"]["n_estimators"]
    100
    >>> from src.config import FEATURE_COLS
    >>> FEATURE_COLS
    ["login_count", ...]
"""

import os
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def load_config(path: str = _CONFIG_PATH) -> Dict[str, Any]:
    """
    Load and validate the YAML configuration file.

    Args:
        path: Absolute or relative path to config.yaml.

    Returns:
        Nested dictionary mirroring the YAML structure.

    Raises:
        FileNotFoundError: If config.yaml is not found.
        ValueError: If the YAML is malformed or missing required keys.

    Example:
        >>> cfg = load_config("config.yaml")
        >>> cfg["model"]["n_estimators"]
        100
    """
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed. Run: pip install pyyaml")
        raise

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Configuration file not found at '{path}'. "
            "Ensure config.yaml exists in the project root."
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Dict[str, Any]) -> None:
    """
    Ensure all required top-level keys exist in the config.

    Args:
        cfg: Loaded configuration dictionary.

    Raises:
        ValueError: If any required section is missing.
    """
    required_keys = ["paths", "preprocessing", "model", "risk_scoring",
                     "attack_simulation", "features"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(
            f"Config file is missing required sections: {missing}. "
            "Check config.yaml against the template."
        )


# -- Singleton loaded once at import time --
settings: Dict[str, Any] = load_config()

# -- Convenience constants (read from settings) --
FEATURE_COLS: List[str] = settings["features"]["feature_cols"]

# Paths
LOGON_PATH: str = settings["paths"]["data"]["logon"]
DEVICE_PATH: str = settings["paths"]["data"]["device"]
FEATURE_PATH: str = settings["paths"]["output"]["feature_table"]
SCORED_PATH: str = settings["paths"]["output"]["scored_results"]
RISK_REPORT_PATH: str = settings["paths"]["output"]["risk_report"]
MODEL_SAVE_PATH: str = settings["paths"]["output"]["model"]
SCALER_SAVE_PATH: str = settings["paths"]["output"]["scaler"]
METADATA_PATH: str = settings["paths"]["output"]["metadata"]
SIM_OUTPUT_PATH: str = settings["paths"]["output"]["simulation_results"]
SIM_REPORT_PATH: str = settings["paths"]["output"]["simulation_report"]

# Preprocessing
OFF_HOUR_START: int = settings["preprocessing"]["off_hour_start"]
OFF_HOUR_END: int = settings["preprocessing"]["off_hour_end"]

# Model
CONTAMINATION_DEFAULT: float = settings["model"]["contamination_default"]
CONTAMINATION_CANDIDATES: List[float] = settings["model"]["contamination_candidates"]
N_ESTIMATORS: int = settings["model"]["n_estimators"]
RANDOM_STATE: int = settings["model"]["random_state"]

# Risk scoring thresholds
THRESHOLD_HIGH: float = settings["risk_scoring"]["threshold_high"]
THRESHOLD_MEDIUM: float = settings["risk_scoring"]["threshold_medium"]
FLAGS: Dict[str, Any] = settings["risk_scoring"]["flags"]

# Attack simulation rule weights
RULE_WEIGHTS: Dict[str, int] = settings["attack_simulation"]["rule_weights"]
TOTAL_RULE_WEIGHT: int = sum(RULE_WEIGHTS.values())
