"""
utmosv2-exported: Standalone inference for UTMOSv2 speech quality model.

This package provides inference for the UTMOSv2 model without requiring
HuggingFace transformers or torchvision dependencies. The SSL encoder
(wav2vec2) and fusion model are both loaded as TorchScript models.

Example:
    >>> from utmosv2_exported import UTMOSv2Predictor
    >>> predictor = UTMOSv2Predictor()
    >>> score = predictor.predict("audio.wav")
    >>> print(f"MOS score: {score:.2f}")
"""

from .config import DEFAULT_CONFIG
from .download import download_models, get_cache_dir, get_model_paths
from .predictor import UTMOSv2Predictor
from .preprocessing import SpectrogramProcessor, load_audio

__version__ = "0.1.0"
__all__ = [
    "UTMOSv2Predictor",
    "DEFAULT_CONFIG",
    "SpectrogramProcessor",
    "load_audio",
    "download_models",
    "get_model_paths",
    "get_cache_dir",
]
