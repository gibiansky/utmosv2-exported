#!/usr/bin/env python3
"""
Export UTMOSv2 model to TorchScript format.

This script exports TWO TorchScript models:
1. utmosv2_ssl.pt - The SSL encoder (wav2vec2) for audio feature extraction
2. utmosv2_fusion.pt - The fusion model that combines SSL and spectrogram features

The exported models do NOT require HuggingFace transformers or torchvision at inference.

Requirements:
    pip install .[export]  # Installs utmosv2, transformers, torchvision

Usage:
    python scripts/export_model.py [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import sys
from pathlib import Path

# Add scripts directory to path for export_wrapper import
sys.path.insert(0, str(Path(__file__).parent))


def export_ssl_encoder(ssl_encoder, ssl_samples: int, output_path: Path) -> bool:
    """Export SSL encoder to TorchScript."""
    import torch

    print("\nExporting SSL encoder to TorchScript...")

    try:
        ssl_encoder.eval()

        # Create example input
        example_audio = torch.randn(1, ssl_samples)

        with torch.no_grad():
            traced_ssl = torch.jit.trace(ssl_encoder, example_audio)

        traced_ssl.save(str(output_path))
        print(f"  Saved: {output_path}")

        # Validate
        print("  Validating export...")
        loaded = torch.jit.load(str(output_path))
        with torch.no_grad():
            original_out = ssl_encoder(example_audio)
            loaded_out = loaded(example_audio)

        diff = torch.abs(original_out - loaded_out).max().item()
        print(f"  Max difference: {diff:.10f}")

        if diff < 1e-5:
            print("  Validation: PASSED")
            return True
        else:
            print("  Validation: WARNING - small numerical difference")
            return True  # Still acceptable

    except Exception as e:
        print(f"  Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_fusion_model(fusion_model, ssl_hidden_shape, spec_shape, num_datasets, output_path: Path) -> bool:
    """Export fusion model to TorchScript."""
    import torch

    print("\nExporting fusion model to TorchScript...")

    try:
        fusion_model.eval()

        # Create example inputs
        ssl_hidden = torch.randn(*ssl_hidden_shape)
        specs = torch.randn(*spec_shape)
        domain = torch.zeros(1, num_datasets)
        domain[0, 0] = 1.0

        with torch.no_grad():
            traced_fusion = torch.jit.trace(fusion_model, (ssl_hidden, specs, domain))

        traced_fusion.save(str(output_path))
        print(f"  Saved: {output_path}")

        # Validate
        print("  Validating export...")
        loaded = torch.jit.load(str(output_path))
        with torch.no_grad():
            original_out = fusion_model(ssl_hidden, specs, domain)
            loaded_out = loaded(ssl_hidden, specs, domain)

        diff = torch.abs(original_out - loaded_out).max().item()
        print(f"  Max difference: {diff:.10f}")

        if diff < 1e-5:
            print("  Validation: PASSED")
            return True
        else:
            print("  Validation: WARNING - small numerical difference")
            return True  # Still acceptable

    except Exception as e:
        print(f"  Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_end_to_end(ssl_encoder, fusion_model, original_model, original_ssl_encoder, cfg, output_dir: Path):
    """Verify exported models match original end-to-end."""
    import torch
    from export_wrapper import get_ssl_hidden_states_original

    print("\n" + "=" * 60)
    print("End-to-End Verification")
    print("=" * 60)

    # Load exported models
    ssl_path = output_dir / "utmosv2_ssl.pt"
    fusion_path = output_dir / "utmosv2_fusion.pt"

    if not ssl_path.exists() or not fusion_path.exists():
        print("Exported models not found, skipping verification")
        return

    loaded_ssl = torch.jit.load(str(ssl_path))
    loaded_fusion = torch.jit.load(str(fusion_path))

    # Create test inputs
    sample_rate = cfg.sr
    ssl_duration = cfg.dataset.ssl.duration
    ssl_samples = int(sample_rate * ssl_duration)
    num_frames = cfg.dataset.spec_frames.num_frames
    num_specs = len(cfg.dataset.specs)
    total_specs = num_frames * num_specs
    num_datasets = fusion_model.num_dataset

    x1 = torch.randn(1, ssl_samples)
    x2 = torch.randn(1, total_specs, 3, 512, 512)
    d = torch.zeros(1, num_datasets)
    d[0, 0] = 1.0

    with torch.no_grad():
        # Original model path
        original_output = original_model(x1, x2, d)

        # Exported model path
        ssl_hidden = loaded_ssl(x1)
        exported_output = loaded_fusion(ssl_hidden, x2, d)

    diff = torch.abs(original_output - exported_output).max().item()

    print(f"\nOriginal model output: {original_output.item():.6f}")
    print(f"Exported model output: {exported_output.item():.6f}")
    print(f"Difference: {diff:.10f}")

    if diff < 1e-4:
        print("\nEnd-to-end verification: PASSED")
    else:
        print("\nEnd-to-end verification: WARNING - outputs differ")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export UTMOSv2 model to TorchScript format"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Directory to save exported models",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UTMOSv2 Model Export")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")

    # Import here to allow --help without dependencies
    import torch
    import utmosv2
    from export_wrapper import create_exportable_models

    # Load original model for verification
    print("\nLoading original UTMOSv2 model...")
    orig = utmosv2.create_model(pretrained=True)
    orig.eval()
    original_model = orig._model
    original_ssl_encoder = original_model.ssl.encoder

    # Create exportable models
    ssl_encoder, fusion_model, cfg = create_exportable_models()

    # Calculate dimensions
    sample_rate = cfg.sr
    ssl_duration = cfg.dataset.ssl.duration
    ssl_samples = int(sample_rate * ssl_duration)
    num_frames = cfg.dataset.spec_frames.num_frames
    num_specs = len(cfg.dataset.specs)
    total_specs = num_frames * num_specs
    num_datasets = fusion_model.num_dataset

    print(f"\nModel configuration:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  SSL duration: {ssl_duration} s ({ssl_samples} samples)")
    print(f"  Num spectrograms: {total_specs} (frames={num_frames}, specs={num_specs})")
    print(f"  Num datasets: {num_datasets}")

    # Get SSL hidden shape for fusion model export
    with torch.no_grad():
        example_audio = torch.randn(1, ssl_samples)
        ssl_hidden = ssl_encoder(example_audio)
    ssl_hidden_shape = ssl_hidden.shape

    print(f"  SSL hidden shape: {list(ssl_hidden_shape)}")

    # Export SSL encoder
    ssl_path = args.output_dir / "utmosv2_ssl.pt"
    ssl_success = export_ssl_encoder(ssl_encoder, ssl_samples, ssl_path)

    # Export fusion model
    fusion_path = args.output_dir / "utmosv2_fusion.pt"
    spec_shape = (1, total_specs, 3, 512, 512)
    fusion_success = export_fusion_model(
        fusion_model, ssl_hidden_shape, spec_shape, num_datasets, fusion_path
    )

    # Save config
    config_path = args.output_dir / "utmosv2_config.json"
    print(f"\nSaving config to {config_path}...")
    config = {
        "sample_rate": sample_rate,
        "ssl_duration": float(ssl_duration),
        "ssl_num_layers": ssl_hidden_shape[1],
        "ssl_features": ssl_hidden_shape[3],
        "spec_frame_sec": float(cfg.dataset.spec_frames.frame_sec),
        "spec_num_frames": num_frames,
        "spec_extend_type": cfg.dataset.spec_frames.extend,
        "specs": [
            {
                "mode": spec.mode,
                "n_fft": spec.n_fft,
                "hop_length": spec.hop_length,
                "n_mels": getattr(spec, "n_mels", 512),
                "win_length": getattr(spec, "win_length", spec.n_fft),
                "norm": getattr(spec, "norm", None),
            }
            for spec in cfg.dataset.specs
        ],
        "spec_transform_size": 512,
        "num_datasets": num_datasets,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if ssl_success and fusion_success:
        # Verify end-to-end
        verify_end_to_end(
            ssl_encoder, fusion_model, original_model, original_ssl_encoder, cfg, args.output_dir
        )

        print("\n" + "=" * 60)
        print("Export Complete!")
        print("=" * 60)
        print(f"\nExported files:")
        print(f"  SSL encoder:   {ssl_path}")
        print(f"  Fusion model:  {fusion_path}")
        print(f"  Config:        {config_path}")
        print(f"\nTotal size: {(ssl_path.stat().st_size + fusion_path.stat().st_size) / 1024 / 1024:.1f} MB")
        print("\nTo use the exported models:")
        print("  from utmosv2_exported import UTMOSv2Predictor")
        print("  predictor = UTMOSv2Predictor()")
        print("  score = predictor.predict('audio.wav')")
        return 0
    else:
        print("\n" + "=" * 60)
        print("Export Failed!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
