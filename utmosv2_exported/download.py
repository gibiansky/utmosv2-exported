"""
Model download utilities for UTMOSv2.

Uses the same cache directory conventions as torch.hub:
- TORCH_HOME environment variable (if set)
- Otherwise ~/.cache/torch/hub

You can override the cache directory by passing cache_dir to the functions,
or by setting the UTMOSV2_CACHE_DIR environment variable.
"""

import hashlib
import os
import urllib.request
from pathlib import Path

from .config import (
    CONFIG_FILENAME,
    FUSION_MODEL_FILENAME,
    GITHUB_RELEASE_URL,
    SSL_MODEL_FILENAME,
)


def get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """
    Get the cache directory for model files.

    Priority:
    1. Explicit cache_dir parameter
    2. UTMOSV2_CACHE_DIR environment variable
    3. TORCH_HOME/hub environment variable
    4. ~/.cache/torch/hub (default)

    Args:
        cache_dir: Optional explicit cache directory

    Returns:
        Path to cache directory (created if doesn't exist)
    """
    if cache_dir is not None:
        cache_path = Path(cache_dir)
    elif "UTMOSV2_CACHE_DIR" in os.environ:
        cache_path = Path(os.environ["UTMOSV2_CACHE_DIR"])
    elif "TORCH_HOME" in os.environ:
        cache_path = Path(os.environ["TORCH_HOME"]) / "hub"
    elif "TORCH_HUB" in os.environ:
        cache_path = Path(os.environ["TORCH_HUB"])
    else:
        cache_path = Path.home() / ".cache" / "torch" / "hub"

    # Add utmosv2-exported subdirectory
    cache_path = cache_path / "utmosv2-exported"
    cache_path.mkdir(parents=True, exist_ok=True)

    return cache_path


def _download_file(url: str, destination: Path, show_progress: bool = True) -> None:
    """Download a file from URL to destination with optional progress."""
    if show_progress:
        print(f"Downloading {url}...")

    # Create a request with a reasonable timeout
    request = urllib.request.Request(url, headers={"User-Agent": "utmosv2-exported"})

    with urllib.request.urlopen(request, timeout=30) as response:
        total_size = response.headers.get("Content-Length")
        if total_size:
            total_size = int(total_size)

        # Download in chunks
        chunk_size = 8192
        downloaded = 0

        with open(destination, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if show_progress and total_size:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(
                        f"\r  {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)",
                        end="",
                        flush=True,
                    )

        if show_progress:
            print()  # Newline after progress


def download_models(
    cache_dir: str | Path | None = None,
    force: bool = False,
    show_progress: bool = True,
) -> tuple[Path, Path, Path]:
    """
    Download UTMOSv2 model files if not already cached.

    Args:
        cache_dir: Optional cache directory override
        force: If True, re-download even if files exist
        show_progress: If True, show download progress

    Returns:
        Tuple of (ssl_model_path, fusion_model_path, config_path)

    Raises:
        RuntimeError: If download fails
    """
    cache_path = get_cache_dir(cache_dir)

    ssl_path = cache_path / SSL_MODEL_FILENAME
    fusion_path = cache_path / FUSION_MODEL_FILENAME
    config_path = cache_path / CONFIG_FILENAME

    files_to_download = [
        (f"{GITHUB_RELEASE_URL}/{SSL_MODEL_FILENAME}", ssl_path),
        (f"{GITHUB_RELEASE_URL}/{FUSION_MODEL_FILENAME}", fusion_path),
        (f"{GITHUB_RELEASE_URL}/{CONFIG_FILENAME}", config_path),
    ]

    for url, dest in files_to_download:
        if dest.exists() and not force:
            if show_progress:
                print(f"Using cached {dest.name}")
            continue

        try:
            _download_file(url, dest, show_progress)
        except Exception as e:
            # Clean up partial download
            if dest.exists():
                dest.unlink()
            raise RuntimeError(f"Failed to download {url}: {e}") from e

    return ssl_path, fusion_path, config_path


def get_model_paths(
    cache_dir: str | Path | None = None,
    auto_download: bool = True,
    show_progress: bool = True,
) -> tuple[Path, Path, Path]:
    """
    Get paths to model files, downloading if necessary.

    Args:
        cache_dir: Optional cache directory override
        auto_download: If True, download models if not present
        show_progress: If True, show download progress

    Returns:
        Tuple of (ssl_model_path, fusion_model_path, config_path)

    Raises:
        FileNotFoundError: If models not found and auto_download is False
        RuntimeError: If download fails
    """
    cache_path = get_cache_dir(cache_dir)

    ssl_path = cache_path / SSL_MODEL_FILENAME
    fusion_path = cache_path / FUSION_MODEL_FILENAME
    config_path = cache_path / CONFIG_FILENAME

    # Check if all files exist
    all_exist = ssl_path.exists() and fusion_path.exists() and config_path.exists()

    if not all_exist:
        if auto_download:
            return download_models(cache_dir, force=False, show_progress=show_progress)
        else:
            missing = []
            if not ssl_path.exists():
                missing.append(SSL_MODEL_FILENAME)
            if not fusion_path.exists():
                missing.append(FUSION_MODEL_FILENAME)
            if not config_path.exists():
                missing.append(CONFIG_FILENAME)
            raise FileNotFoundError(
                f"Model files not found: {', '.join(missing)}. "
                f"Set auto_download=True or run download_models() first."
            )

    return ssl_path, fusion_path, config_path
