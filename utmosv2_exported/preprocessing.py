"""
Audio preprocessing for UTMOSv2 inference.

This module handles:
- Audio loading and resampling
- Mel spectrogram computation
- Audio segment selection

IMPORTANT: This module does NOT use transformers or torchvision.
- Spectrogram resizing uses torch.nn.functional.interpolate
- SSL encoding is done via a separate TorchScript model
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from .config import DEFAULT_CONFIG


class SpectrogramProcessor:
    """
    Standalone spectrogram processor replicating UTMOSv2's preprocessing.

    Uses librosa for mel spectrogram computation (matching UTMOSv2 exactly)
    and torch.nn.functional.interpolate for resizing (no torchvision).
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize the spectrogram processor.

        Args:
            config: Configuration dict. Uses DEFAULT_CONFIG if not provided.
        """
        self.config = config or DEFAULT_CONFIG
        self.sample_rate = self.config["sample_rate"]
        self.frame_samples = int(self.config["spec_frame_sec"] * self.sample_rate)
        self.num_frames = self.config["spec_num_frames"]
        self.specs = self.config["specs"]
        self.extend_type = self.config["spec_extend_type"]
        self.transform_size = self.config["spec_transform_size"]

    def _extend_audio(self, audio: torch.Tensor, target_length: int) -> torch.Tensor:
        """Extend audio to target length using tiling."""
        if len(audio) >= target_length:
            return audio

        if self.extend_type == "tile":
            # Tile the audio to reach target length
            repeats = (target_length // len(audio)) + 1
            audio = audio.repeat(repeats)

        return audio[:target_length]

    def _make_melspec(self, audio: torch.Tensor, spec_config: dict) -> torch.Tensor:
        """Compute mel spectrogram matching UTMOSv2's librosa-based implementation."""
        import librosa

        n_fft = spec_config["n_fft"]
        hop_length = spec_config["hop_length"]
        n_mels = spec_config["n_mels"]
        win_length = spec_config["win_length"]
        norm_val = spec_config["norm"]

        # Convert to numpy for librosa (matches UTMOSv2 exactly)
        audio_np = audio.numpy()

        # Compute mel spectrogram using librosa (exactly like UTMOSv2)
        spec = librosa.feature.melspectrogram(
            y=audio_np,
            sr=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            win_length=win_length,
        )

        # Convert to dB (like UTMOSv2)
        spec = librosa.power_to_db(spec, ref=np.max)

        # Normalize (matches UTMOSv2's (spec + norm) / norm)
        if norm_val is not None:
            spec = (spec + norm_val) / norm_val

        return torch.from_numpy(spec.astype(np.float32))

    def _resize_spec(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Resize spectrogram to target size using torch interpolation.

        This replaces torchvision.transforms.Resize with
        torch.nn.functional.interpolate.
        """
        # spec is (3, H, W), interpolate expects (N, C, H, W)
        spec = spec.unsqueeze(0)  # (1, 3, H, W)
        spec = F.interpolate(
            spec,
            size=(self.transform_size, self.transform_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return spec.squeeze(0)  # (3, H, W)

    def _select_random_start(
        self, audio: torch.Tensor, length: int, rng: np.random.Generator | None = None
    ) -> torch.Tensor:
        """Select a random segment from audio (matches UTMOSv2 behavior)."""
        if len(audio) <= length:
            return audio
        if rng is not None:
            start = rng.integers(0, len(audio) - length)
        else:
            start = np.random.randint(0, len(audio) - length)
        return audio[start : start + length]

    def process(
        self, audio: torch.Tensor, rng: np.random.Generator | None = None
    ) -> torch.Tensor:
        """
        Process audio into stacked spectrograms.

        Args:
            audio: Audio tensor (samples,) at correct sample rate
            rng: Optional numpy random generator for reproducibility

        Returns:
            Stacked spectrograms (num_frames * num_specs, 3, 512, 512)
        """
        # Extend audio if needed (matches UTMOSv2's extend_audio)
        audio = self._extend_audio(audio, self.frame_samples)

        specs = []
        for _ in range(self.num_frames):
            # Select random segment (matches UTMOSv2's select_random_start)
            frame_audio = self._select_random_start(audio, self.frame_samples, rng)

            for spec_config in self.specs:
                # Compute mel spectrogram
                spec = self._make_melspec(frame_audio, spec_config)

                # Stack to 3 channels (RGB-like)
                spec = spec.unsqueeze(0).repeat(3, 1, 1)  # (3, n_mels, time)

                # Resize to target size using torch interpolate (no torchvision!)
                spec = self._resize_spec(spec)

                specs.append(spec)

        # Stack all spectrograms
        return torch.stack(specs, dim=0)  # (num_frames * num_specs, 3, 512, 512)


def load_audio(
    audio_path: str, sample_rate: int = 16000
) -> torch.Tensor:
    """
    Load and preprocess audio file.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default 16000)

    Returns:
        Audio tensor at correct sample rate, shape (samples,)
    """
    import soundfile as sf

    # Load audio with soundfile (more reliable than torchaudio for various formats)
    waveform, sr = sf.read(str(audio_path), dtype="float32")

    # Convert to torch tensor
    waveform = torch.from_numpy(waveform)

    # Handle stereo -> mono
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    return waveform


def prepare_ssl_input(
    audio: torch.Tensor, ssl_samples: int, rng: np.random.Generator | None = None
) -> torch.Tensor:
    """
    Prepare audio for SSL encoder input.

    Args:
        audio: Audio tensor (samples,)
        ssl_samples: Number of samples required for SSL
        rng: Optional numpy random generator for reproducibility

    Returns:
        Audio tensor of exactly ssl_samples length
    """
    # Extend audio if needed
    if len(audio) < ssl_samples:
        repeats = (ssl_samples // len(audio)) + 1
        audio = audio.repeat(repeats)

    # Select random segment
    if len(audio) > ssl_samples:
        if rng is not None:
            start = rng.integers(0, len(audio) - ssl_samples)
        else:
            start = np.random.randint(0, len(audio) - ssl_samples)
        audio = audio[start : start + ssl_samples]
    else:
        audio = audio[:ssl_samples]

    return audio
