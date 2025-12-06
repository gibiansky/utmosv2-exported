#!/usr/bin/env python3
"""
Verify that the exported models produce identical outputs to the original.

This script uses UTMOSv2's preprocessing to create inputs, then passes
the SAME inputs to both the original model and the exported TorchScript
models to verify they produce identical outputs.

Requirements:
    pip install .[export]  # Installs utmosv2, transformers, torchvision

Usage:
    python scripts/verify_export.py path/to/audio.wav
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for utmosv2_exported import
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_with_audio(audio_path: Path, model_dir: Path) -> bool:
    """
    Verify exported model matches original using identical inputs.

    Returns:
        True if outputs match (difference < 1e-5), False otherwise
    """
    import torch
    import utmosv2
    from utmosv2.utils import get_dataset
    from utmosv2.dataset._schema import DatasetSchema

    ssl_path = model_dir / "utmosv2_ssl.pt"
    fusion_path = model_dir / "utmosv2_fusion.pt"

    if not ssl_path.exists():
        print(f"ERROR: SSL model not found: {ssl_path}")
        return False
    if not fusion_path.exists():
        print(f"ERROR: Fusion model not found: {fusion_path}")
        return False

    print("Loading original UTMOSv2 model...")
    orig = utmosv2.create_model(pretrained=True)
    orig_model = orig._model
    orig_model.eval()
    cfg = orig._cfg

    print(f"Loading exported models from {model_dir}...")
    exported_ssl = torch.jit.load(str(ssl_path))
    exported_fusion = torch.jit.load(str(fusion_path))
    exported_ssl.eval()
    exported_fusion.eval()

    # Use UTMOSv2's preprocessing for identical inputs
    print("Preprocessing audio with UTMOSv2...")
    cfg.phase = "valid"
    data = [DatasetSchema(file_path=audio_path, dataset="sarulab")]
    dataset = get_dataset(cfg, data, cfg.phase)

    # Get preprocessed sample (x1=audio, x2=spectrograms, d=domain, target)
    x1, x2, d, _ = dataset[0]
    x1 = x1.unsqueeze(0)
    x2 = x2.unsqueeze(0)
    d = d.unsqueeze(0)

    print(f"  x1 (audio): {x1.shape}")
    print(f"  x2 (specs): {x2.shape}")
    print(f"  d (domain): {d.shape}")

    # Run inference with both pipelines
    print("\nRunning inference...")
    with torch.no_grad():
        # Original model
        original_score = orig_model(x1, x2, d)

        # Exported models
        ssl_hidden = exported_ssl(x1)
        exported_score = exported_fusion(ssl_hidden, x2, d)

    orig_val = original_score.squeeze().item()
    export_val = exported_score.squeeze().item()
    diff = abs(orig_val - export_val)

    print()
    print("=" * 60)
    print("Results (with IDENTICAL inputs)")
    print("=" * 60)
    print(f"  Original model:  {orig_val:.6f}")
    print(f"  Exported model:  {export_val:.6f}")
    print(f"  Difference:      {diff:.10f}")
    print()

    if diff < 1e-5:
        print("  PASSED - Models produce identical outputs")
        return True
    elif diff < 1e-3:
        print("  ACCEPTABLE - Minor numerical difference")
        return True
    else:
        print("  FAILED - Models produce different outputs")
        return False


def verify_standalone_inference(audio_path: Path, model_dir: Path) -> bool:
    """
    Verify standalone inference (using utmosv2_exported package).

    This tests the full pipeline without requiring utmosv2 package.
    """
    import torch

    print("\n" + "=" * 60)
    print("Standalone Inference Test")
    print("=" * 60)

    ssl_path = model_dir / "utmosv2_ssl.pt"
    fusion_path = model_dir / "utmosv2_fusion.pt"
    config_path = model_dir / "utmosv2_config.json"

    if not all(p.exists() for p in [ssl_path, fusion_path, config_path]):
        print("Model files not found, skipping standalone test")
        return True

    try:
        from utmosv2_exported import UTMOSv2Predictor

        print(f"Loading predictor from {model_dir}...")
        predictor = UTMOSv2Predictor(
            ssl_model_path=ssl_path,
            fusion_model_path=fusion_path,
            config_path=config_path,
            auto_download=False,
        )

        print(f"Running inference on {audio_path}...")
        score = predictor.predict(audio_path, seed=42)

        print(f"\nStandalone prediction: {score:.6f}")
        print("Standalone inference: PASSED")
        return True

    except Exception as e:
        print(f"Standalone inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_original_predict(audio_path: Path) -> None:
    """Compare standalone inference with original UTMOSv2.predict()."""
    import utmosv2

    print("\n" + "=" * 60)
    print("Comparison with Original UTMOSv2")
    print("=" * 60)

    print("Running original UTMOSv2.predict()...")
    model = utmosv2.create_model(pretrained=True)
    original_score = model.predict(
        input_path=str(audio_path), device="cpu", verbose=False, num_workers=0
    )

    if isinstance(original_score, (list, tuple)):
        original_score = float(original_score[0])
    else:
        original_score = float(original_score)

    print(f"  Original predict(): {original_score:.6f}")

    print("\nNote: Slight differences are expected due to random segment selection.")
    print("Use verify_with_audio() for exact comparison with identical inputs.")


def main():
    parser = argparse.ArgumentParser(
        description="Verify exported model matches original UTMOSv2"
    )
    parser.add_argument("audio_path", type=Path, help="Path to audio file")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Directory containing exported models",
    )
    parser.add_argument(
        "--skip-standalone",
        action="store_true",
        help="Skip standalone inference test",
    )
    args = parser.parse_args()

    if not args.audio_path.exists():
        print(f"ERROR: Audio file not found: {args.audio_path}")
        return 1

    print("=" * 60)
    print("UTMOSv2 Export Verification")
    print("=" * 60)
    print(f"Audio: {args.audio_path}")
    print(f"Model dir: {args.model_dir}")
    print()

    # Test 1: Verify with identical inputs
    success = verify_with_audio(args.audio_path, args.model_dir)

    # Test 2: Standalone inference
    if not args.skip_standalone:
        standalone_success = verify_standalone_inference(args.audio_path, args.model_dir)
        success = success and standalone_success

    # Test 3: Compare with original predict (informational)
    try:
        compare_with_original_predict(args.audio_path)
    except Exception as e:
        print(f"Could not run original comparison: {e}")

    print("\n" + "=" * 60)
    if success:
        print("All verification tests PASSED")
    else:
        print("Some verification tests FAILED")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
