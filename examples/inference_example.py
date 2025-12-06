#!/usr/bin/env python3
"""
Example usage of utmosv2-exported for speech quality assessment.

This script demonstrates:
1. Basic single-file prediction
2. Batch prediction
3. Direct tensor prediction
4. Custom model paths
"""

from pathlib import Path


def example_basic():
    """Basic usage - predict MOS score for a single audio file."""
    from utmosv2_exported import UTMOSv2Predictor

    # Create predictor (downloads models automatically on first use)
    predictor = UTMOSv2Predictor()

    # Predict MOS score
    score = predictor.predict("audio.wav")
    print(f"MOS score: {score:.2f}")


def example_with_seed():
    """Reproducible prediction with fixed seed."""
    from utmosv2_exported import UTMOSv2Predictor

    predictor = UTMOSv2Predictor()

    # Use seed for reproducible results (affects random segment selection)
    score1 = predictor.predict("audio.wav", seed=42)
    score2 = predictor.predict("audio.wav", seed=42)

    print(f"Score 1: {score1:.4f}")
    print(f"Score 2: {score2:.4f}")
    assert score1 == score2, "Scores should be identical with same seed"


def example_batch():
    """Batch prediction for multiple files."""
    from utmosv2_exported import UTMOSv2Predictor

    predictor = UTMOSv2Predictor()

    audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
    scores = predictor.predict_batch(audio_files)

    for path, score in zip(audio_files, scores):
        print(f"{path}: {score:.2f}")


def example_tensor():
    """Direct tensor prediction (useful for pipelines)."""
    import torch
    import torchaudio

    from utmosv2_exported import UTMOSv2Predictor

    predictor = UTMOSv2Predictor()

    # Load audio as tensor
    waveform, sr = torchaudio.load("audio.wav")

    # Convert to mono and resample to 16kHz if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # Predict from tensor
    score = predictor.predict_tensor(waveform, seed=42)
    print(f"MOS score: {score:.2f}")


def example_custom_paths():
    """Use custom model paths (e.g., for local models)."""
    from utmosv2_exported import UTMOSv2Predictor

    predictor = UTMOSv2Predictor(
        ssl_model_path="/path/to/utmosv2_ssl.pt",
        fusion_model_path="/path/to/utmosv2_fusion.pt",
        config_path="/path/to/utmosv2_config.json",
        device="cuda",  # Use GPU
        auto_download=False,  # Don't try to download
    )

    score = predictor.predict("audio.wav")
    print(f"MOS score: {score:.2f}")


def example_gpu():
    """Use GPU for faster inference."""
    import torch

    from utmosv2_exported import UTMOSv2Predictor

    # Check GPU availability
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
    else:
        device = "cpu"

    print(f"Using device: {device}")

    predictor = UTMOSv2Predictor(device=device)
    score = predictor.predict("audio.wav")
    print(f"MOS score: {score:.2f}")


def main():
    """Run examples (requires audio files to exist)."""
    print("UTMOSv2 Exported - Usage Examples")
    print("=" * 50)

    # Check if we have a test audio file
    test_audio = Path("audio.wav")
    if not test_audio.exists():
        print("\nNote: Create 'audio.wav' to run these examples")
        print("\nExample code snippets:")
        print()

        print("# Basic usage")
        print("from utmosv2_exported import UTMOSv2Predictor")
        print("predictor = UTMOSv2Predictor()")
        print("score = predictor.predict('audio.wav')")
        print()

        print("# Batch prediction")
        print("scores = predictor.predict_batch(['a.wav', 'b.wav'])")
        print()

        print("# GPU inference")
        print("predictor = UTMOSv2Predictor(device='cuda')")
        return

    print("\n1. Basic prediction:")
    example_basic()

    print("\n2. Reproducible prediction:")
    example_with_seed()

    print("\n3. GPU inference:")
    example_gpu()


if __name__ == "__main__":
    main()
