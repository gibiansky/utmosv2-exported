"""
Main predictor class for UTMOSv2 inference.

This module provides the UTMOSv2Predictor class, which handles all inference
using only torch, torchaudio, soundfile, and librosa - NO transformers or
torchvision dependencies.

The SSL encoder (wav2vec2) is loaded as a TorchScript model, eliminating
the need for HuggingFace transformers.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_CONFIG
from .download import get_model_paths
from .preprocessing import SpectrogramProcessor, load_audio, prepare_ssl_input


class UTMOSv2Predictor:
    """
    Standalone UTMOSv2 predictor for speech quality assessment.

    This predictor uses two TorchScript models:
    - SSL model: wav2vec2 encoder exported to TorchScript
    - Fusion model: Main UTMOSv2 fusion model

    No HuggingFace transformers or torchvision dependencies required.

    Example:
        >>> predictor = UTMOSv2Predictor()
        >>> score = predictor.predict("audio.wav")
        >>> print(f"MOS score: {score:.2f}")

        >>> # Batch prediction
        >>> scores = predictor.predict_batch(["audio1.wav", "audio2.wav"])
    """

    def __init__(
        self,
        ssl_model_path: str | Path | None = None,
        fusion_model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        device: str = "cpu",
        auto_download: bool = True,
    ):
        """
        Initialize the predictor.

        Args:
            ssl_model_path: Path to SSL TorchScript model. If None, downloads/uses cached.
            fusion_model_path: Path to fusion TorchScript model. If None, downloads/uses cached.
            config_path: Path to config JSON. If None, uses default config.
            cache_dir: Cache directory for downloaded models. Uses TORCH_HOME if not set.
            device: Device to run inference on ("cpu", "cuda", "mps").
            auto_download: If True, automatically download models if not present.
        """
        self.device = torch.device(device)

        # Get model paths (download if needed)
        if ssl_model_path is None or fusion_model_path is None:
            ssl_path, fusion_path = get_model_paths(
                cache_dir=cache_dir,
                auto_download=auto_download,
                show_progress=True,
            )
            ssl_model_path = ssl_model_path or ssl_path
            fusion_model_path = fusion_model_path or fusion_path

        # Load config (use bundled config by default, or custom if provided)
        if config_path is not None and Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = DEFAULT_CONFIG

        # Load TorchScript models
        # Note: We load to CPU first, then move to target device. This is necessary
        # because: (1) map_location only affects parameter loading, but traced
        # constants (like torch.zeros calls) may have device hardcoded, and
        # (2) some devices like MPS have dtype restrictions during loading.
        # Calling .to(device) moves all tensors including traced constants.
        self.ssl_model = torch.jit.load(str(ssl_model_path), map_location="cpu")
        self.ssl_model.to(self.device)
        self.ssl_model.eval()

        self.fusion_model = torch.jit.load(str(fusion_model_path), map_location="cpu")
        self.fusion_model.to(self.device)
        self.fusion_model.eval()

        # Initialize spectrogram processor
        self.spec_processor = SpectrogramProcessor(self.config)

        # SSL parameters
        self.ssl_samples = int(
            self.config["ssl_duration"] * self.config["sample_rate"]
        )
        self.sample_rate = self.config["sample_rate"]
        self.num_datasets = self.config["num_datasets"]

    def _preprocess(
        self, audio: torch.Tensor, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess audio for model input.

        Args:
            audio: Audio tensor (samples,) on any device
            seed: Random seed for reproducible segment selection

        Returns:
            Tuple of (ssl_hidden_states, spectrograms, domain_embedding)
        """
        # Ensure audio is on CPU for preprocessing (librosa requires numpy)
        audio = audio.cpu()

        # Create RNG for reproducibility
        rng = np.random.default_rng(seed) if seed is not None else None

        # Prepare audio for SSL encoder
        ssl_audio = prepare_ssl_input(audio, self.ssl_samples, rng)

        # Run SSL encoder
        # Input: (samples,) -> unsqueeze to (1, samples) for batch dimension
        ssl_input = ssl_audio.unsqueeze(0).to(self.device)
        ssl_hidden = self.ssl_model(ssl_input)  # (1, num_layers, seq_len, features)

        # Compute spectrograms
        specs = self.spec_processor.process(audio, rng)  # (num_specs, 3, 512, 512)
        specs = specs.unsqueeze(0).to(self.device)  # (1, num_specs, 3, 512, 512)

        # Create domain embedding (use zeros for generic/unknown domain)
        domain = torch.zeros(1, self.num_datasets, device=self.device)

        return ssl_hidden, specs, domain

    @torch.no_grad()
    def predict(self, audio_path: str | Path, seed: int | None = None) -> float:
        """
        Predict MOS score for an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
            seed: Random seed for reproducible segment selection (optional)

        Returns:
            Predicted MOS score (typically 1-5 scale)
        """
        # Load audio
        audio = load_audio(str(audio_path), self.sample_rate)

        # Preprocess
        ssl_hidden, specs, domain = self._preprocess(audio, seed)

        # Run fusion model
        score = self.fusion_model(ssl_hidden, specs, domain)

        return float(score.squeeze().cpu())

    @torch.no_grad()
    def predict_batch(
        self, audio_paths: list[str | Path], seed: int | None = None
    ) -> list[float]:
        """
        Predict MOS scores for multiple audio files.

        Currently processes sequentially. Batch processing could be optimized
        in the future for files of similar length.

        Args:
            audio_paths: List of paths to audio files
            seed: Base random seed. Each file uses seed+index for reproducibility.

        Returns:
            List of predicted MOS scores
        """
        scores = []
        for i, path in enumerate(audio_paths):
            file_seed = (seed + i) if seed is not None else None
            scores.append(self.predict(path, seed=file_seed))
        return scores

    @torch.no_grad()
    def predict_tensor(
        self, audio: torch.Tensor, seed: int | None = None
    ) -> float:
        """
        Predict MOS score for audio tensor directly.

        Args:
            audio: Audio tensor (samples,) at 16kHz sample rate
            seed: Random seed for reproducible segment selection (optional)

        Returns:
            Predicted MOS score (typically 1-5 scale)
        """
        # Preprocess
        ssl_hidden, specs, domain = self._preprocess(audio, seed)

        # Run fusion model
        score = self.fusion_model(ssl_hidden, specs, domain)

        return float(score.squeeze().cpu())
