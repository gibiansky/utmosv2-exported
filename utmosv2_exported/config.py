"""
Configuration constants for UTMOSv2 inference.

These values match the UTMOSv2 fusion_stage3 model configuration.
"""

import json
from pathlib import Path


def _load_bundled_config() -> dict:
    """Load the bundled config JSON file."""
    config_path = Path(__file__).parent / "utmosv2_config.json"
    with open(config_path) as f:
        return json.load(f)


# Default preprocessing config (loaded from bundled JSON)
DEFAULT_CONFIG: dict = _load_bundled_config()

# Model file names
SSL_MODEL_FILENAME = "utmosv2_ssl.pt"
FUSION_MODEL_FILENAME = "utmosv2_fusion.pt"

# GitHub release URL for model download
GITHUB_RELEASE_URL = (
    "https://github.com/gibiansky/utmosv2-exported/releases/download/v0.1"
)
