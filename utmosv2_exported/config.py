"""
Configuration constants for UTMOSv2 inference.

These values match the UTMOSv2 fusion_stage3 model configuration.
"""

# Default preprocessing config (matches UTMOSv2 fusion_stage3)
DEFAULT_CONFIG: dict = {
    "sample_rate": 16000,
    "ssl_duration": 3.0,  # seconds
    "ssl_num_layers": 13,  # hidden layers from wav2vec2-base
    "ssl_features": 768,  # hidden size from wav2vec2-base
    "spec_frame_sec": 1.4,  # seconds per spectrogram frame
    "spec_num_frames": 2,
    "spec_extend_type": "tile",
    "specs": [
        {
            "mode": "melspec",
            "n_fft": 4096,
            "hop_length": 32,
            "n_mels": 512,
            "win_length": 4096,
            "norm": 80,
        },
        {
            "mode": "melspec",
            "n_fft": 4096,
            "hop_length": 32,
            "n_mels": 512,
            "win_length": 2048,
            "norm": 80,
        },
        {
            "mode": "melspec",
            "n_fft": 4096,
            "hop_length": 32,
            "n_mels": 512,
            "win_length": 1024,
            "norm": 80,
        },
        {
            "mode": "melspec",
            "n_fft": 4096,
            "hop_length": 32,
            "n_mels": 512,
            "win_length": 512,
            "norm": 80,
        },
    ],
    "spec_transform_size": 512,  # Resize spectrograms to 512x512
    "num_datasets": 10,  # Number of domain embeddings
}

# Model file names
SSL_MODEL_FILENAME = "utmosv2_ssl.pt"
FUSION_MODEL_FILENAME = "utmosv2_fusion.pt"
CONFIG_FILENAME = "utmosv2_config.json"

# GitHub release URL for model download
GITHUB_RELEASE_URL = (
    "https://github.com/gibiansky/utmosv2-exported/releases/download/v0.1.0"
)
