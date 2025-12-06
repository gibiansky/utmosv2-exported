#!/usr/bin/env python3
"""
Export-friendly wrapper for UTMOSv2 model.

This module creates TorchScript-compatible versions of:
1. The SSL encoder (wav2vec2) - for audio feature extraction
2. The fusion model - for combining SSL and spectrogram features

IMPORTANT: Requires `pip install .[export]` for transformers and utmosv2.
"""

import torch
import torch.nn as nn


class ExportableSSLEncoder(nn.Module):
    """
    TorchScript-compatible wrapper for wav2vec2 SSL encoder.

    This bypasses the HuggingFace FeatureExtractor (which uses numpy)
    and directly uses the wav2vec2 model with pure PyTorch preprocessing.
    """

    def __init__(self, wav2vec2_model: nn.Module, sample_rate: int = 16000):
        super().__init__()
        self.model = wav2vec2_model
        self.sample_rate = sample_rate

        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Process audio and return stacked hidden states.

        Args:
            audio: Audio tensor (batch, samples) - should be at 16kHz

        Returns:
            Stacked hidden states (batch, num_layers, seq_len, features)
        """
        # Normalize audio (wav2vec2 expects normalized input)
        # The FeatureExtractor normally does this, but we do it in PyTorch
        audio = audio - audio.mean(dim=1, keepdim=True)

        # Get hidden states from wav2vec2
        outputs = self.model(audio, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (batch, seq, features)

        # Stack to (batch, num_layers, seq, features)
        stacked = torch.stack(hidden_states, dim=1)

        return stacked


class ExportableSSLProcessor(nn.Module):
    """
    TorchScript-compatible SSL hidden state processor.

    Takes pre-computed SSL hidden states and processes them with
    proper tensor operations (no iteration over weights).
    """

    def __init__(
        self,
        weights: torch.Tensor,
        attn_layers: nn.ModuleList | None,
        fc_in_features: int,
        num_classes: int,
        use_attn: bool,
    ):
        super().__init__()
        # Store weights as buffer (not parameter, since we're loading trained weights)
        self.register_buffer("weights", weights)
        self.use_attn = use_attn
        self.attn_layers = attn_layers if use_attn else None

        # FC layer will be an Identity since we remove it at the top level
        self.fc = nn.Identity()
        self.fc_in_features = fc_in_features

    def forward(self, hidden_states: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Process SSL hidden states.

        Args:
            hidden_states: Stacked hidden states tensor (batch, num_layers, seq_len, features)
            d: Domain embedding (batch, num_datasets) - concatenated to output as in original

        Returns:
            Processed features (batch, fc_in_features)
        """
        # hidden_states shape: (batch, num_layers, seq_len, features)
        # weights shape: (num_layers,)
        # Output should be: (batch, seq_len, features)

        # Use einsum for weighted sum over layers (no iteration!)
        # 'blsf,l->bsf' : batch, layers, seq, features weighted by layers -> batch, seq, features
        x = torch.einsum("blsf,l->bsf", hidden_states, self.weights)

        if self.use_attn and self.attn_layers is not None:
            y = x
            for attn in self.attn_layers:
                y, _ = attn(y, y, y)
            # Concat mean and max pooling
            x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=1)[0]], dim=1)
        else:
            # Concat mean and max pooling over sequence dimension
            x = torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1)[0]], dim=1)

        # Concatenate domain embedding (matches original behavior when fc is Identity)
        x = torch.cat([x, d], dim=1)

        return x


class ExportableMultiSpecModel(nn.Module):
    """
    TorchScript-compatible multi-spectrogram model.

    Replaces iteration over weights with explicit tensor operations.
    """

    def __init__(
        self,
        backbones: nn.ModuleList,
        weights: torch.Tensor,
        pooling: nn.Module,
        attn: nn.Module | None,
        num_frames: int,
        num_specs: int,
        use_attn: bool,
    ):
        super().__init__()
        self.backbones = backbones
        self.register_buffer("spec_weights", weights)
        self.pooling = pooling
        self.attn = attn
        self.num_frames = num_frames
        self.num_specs = num_specs
        self.use_attn = use_attn
        self.fc = nn.Identity()

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Process spectrograms.

        Args:
            x: Stacked spectrograms (batch, num_frames*num_specs, channels, height, width)
            d: Domain embedding (batch, num_datasets) - unused, kept for interface compatibility

        Returns:
            Processed features
        """
        batch_size = x.shape[0]
        total_specs = self.num_frames * self.num_specs

        # Process each spectrogram through its backbone
        # Use explicit indexing instead of list comprehension
        backbone_outputs = []
        for i in range(total_specs):
            spec_i = x[:, i, :, :, :]  # (batch, channels, height, width)
            backbone_idx = i % self.num_specs
            out = self.backbones[backbone_idx](spec_i)
            backbone_outputs.append(out)

        # Stack for weighted sum: (num_frames, num_specs, batch, features, h, w)
        # First reshape backbone outputs
        frame_outputs = []
        for frame_idx in range(self.num_frames):
            # Get outputs for this frame across all spec types
            frame_specs = []
            for spec_idx in range(self.num_specs):
                idx = frame_idx * self.num_specs + spec_idx
                frame_specs.append(backbone_outputs[idx])

            # Stack and do weighted sum using broadcasting
            # frame_specs is list of (batch, features, h, w)
            stacked = torch.stack(frame_specs, dim=0)  # (num_specs, batch, features, h, w)
            # Weighted sum using einsum
            # 's,sbfhw->bfhw' : specs weighted -> batch, features, h, w
            weighted = torch.einsum("s,sbfhw->bfhw", self.spec_weights, stacked)
            frame_outputs.append(weighted)

        # Concatenate frames along width dimension (dim=3)
        x = torch.cat(frame_outputs, dim=3)

        # Pooling
        x = self.pooling(x).squeeze(3)

        # Attention if configured
        if self.use_attn and self.attn is not None:
            xt = torch.permute(x, (0, 2, 1))
            y, _ = self.attn(xt, xt, xt)
            x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=2).values], dim=1)

        # Concatenate domain embedding (matches original behavior when fc is Identity)
        x = torch.cat([x, d], dim=1)

        return x


class ExportableFusionModel(nn.Module):
    """
    Exportable fusion model (takes SSL hidden states, not raw audio).

    This is the main model that gets exported to TorchScript.
    It takes pre-computed SSL hidden states from the SSL encoder.
    """

    def __init__(
        self,
        ssl_processor: ExportableSSLProcessor,
        spec_model: ExportableMultiSpecModel,
        fc: nn.Linear,
        num_dataset: int,
    ):
        super().__init__()
        self.ssl_processor = ssl_processor
        self.spec_model = spec_model
        self.fc = fc
        self.num_dataset = num_dataset

    def forward(
        self, ssl_hidden_states: torch.Tensor, spectrograms: torch.Tensor, domain: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            ssl_hidden_states: Pre-computed SSL hidden states (batch, num_layers, seq_len, features)
            spectrograms: Stacked spectrograms (batch, num_frames*num_specs, 3, 512, 512)
            domain: Domain one-hot embedding (batch, num_datasets)

        Returns:
            MOS score prediction (batch, 1)
        """
        # Create dummy domain tensors for sub-models (they don't use them after fc removal)
        dummy_d = torch.zeros(
            ssl_hidden_states.shape[0], self.num_dataset, device=ssl_hidden_states.device
        )

        # Process SSL features
        ssl_features = self.ssl_processor(ssl_hidden_states, dummy_d)

        # Process spectrograms
        spec_features = self.spec_model(spectrograms, dummy_d)

        # Concatenate all features
        x = torch.cat([ssl_features, spec_features, domain], dim=1)

        # Final FC layer
        x = self.fc(x)

        return x


def create_exportable_models():
    """
    Create exportable UTMOSv2 models by loading weights from the original.

    Returns:
        Tuple of (ssl_encoder, fusion_model, config)
        - ssl_encoder: ExportableSSLEncoder for wav2vec2 (TorchScript exportable)
        - fusion_model: ExportableFusionModel (TorchScript exportable)
        - config: Model configuration
    """
    import utmosv2

    print("Loading original UTMOSv2 model...")
    orig = utmosv2.create_model(pretrained=True)
    orig.eval()

    # Get the underlying torch model
    torch_model = orig._model
    cfg = orig._cfg

    # Get model components
    ssl_model = torch_model.ssl
    spec_model = torch_model.spec_long

    # Create exportable SSL encoder (wraps wav2vec2)
    print("Creating exportable SSL encoder...")
    ssl_encoder = ExportableSSLEncoder(
        wav2vec2_model=ssl_model.encoder.model,
        sample_rate=cfg.sr,
    )

    # Get SSL config
    use_ssl_attn = cfg.model.ssl.attn > 0

    # Create exportable SSL processor
    ssl_processor = ExportableSSLProcessor(
        weights=ssl_model.weights.data.clone(),
        attn_layers=ssl_model.attn if use_ssl_attn else None,
        fc_in_features=ssl_model.fc.in_features if hasattr(ssl_model.fc, "in_features") else 0,
        num_classes=cfg.model.ssl.num_classes,
        use_attn=use_ssl_attn,
    )

    # Get spec model config
    num_frames = cfg.dataset.spec_frames.num_frames
    num_specs = len(cfg.dataset.specs)
    use_spec_attn = cfg.model.multi_spec.atten

    # Create exportable spec model
    exportable_spec = ExportableMultiSpecModel(
        backbones=spec_model.backbones,
        weights=spec_model.weights.data.clone(),
        pooling=spec_model.pooling,
        attn=spec_model.attn if use_spec_attn else None,
        num_frames=num_frames,
        num_specs=num_specs,
        use_attn=use_spec_attn,
    )

    # Create fusion model
    print("Creating exportable fusion model...")
    fusion_model = ExportableFusionModel(
        ssl_processor=ssl_processor,
        spec_model=exportable_spec,
        fc=torch_model.fc,
        num_dataset=torch_model.num_dataset,
    )
    fusion_model.eval()

    return ssl_encoder, fusion_model, cfg


def get_ssl_hidden_states_original(encoder, audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Get SSL hidden states from raw audio using the ORIGINAL encoder.

    This is for verification only - uses the HuggingFace encoder.

    Args:
        encoder: The original SSL encoder (_SSLEncoder)
        audio_tensor: Raw audio tensor (batch, samples)

    Returns:
        Hidden states tensor (batch, num_layers, seq_len, features)
    """
    # The encoder expects a tuple of tensors
    audio_list = [audio_tensor[i] for i in range(audio_tensor.shape[0])]

    with torch.no_grad():
        hidden_states = encoder(tuple(audio_list))

    # hidden_states is a tuple of (batch, seq_len, features) tensors
    # Stack them to (batch, num_layers, seq_len, features)
    stacked = torch.stack(hidden_states, dim=1)

    return stacked


if __name__ == "__main__":
    print("=" * 60)
    print("Creating Exportable UTMOSv2 Models")
    print("=" * 60)

    ssl_encoder, fusion_model, cfg = create_exportable_models()

    # Create test inputs
    sample_rate = cfg.sr
    ssl_duration = cfg.dataset.ssl.duration
    ssl_samples = int(sample_rate * ssl_duration)
    num_frames = cfg.dataset.spec_frames.num_frames
    num_specs = len(cfg.dataset.specs)
    total_specs = num_frames * num_specs
    num_datasets = fusion_model.num_dataset

    print(f"\nModel configuration:")
    print(f"  SSL samples: {ssl_samples}")
    print(f"  Num spectrograms: {total_specs}")
    print(f"  Num datasets: {num_datasets}")

    x1 = torch.randn(1, ssl_samples)
    x2 = torch.randn(1, total_specs, 3, 512, 512)
    d = torch.zeros(1, num_datasets)
    d[0, 0] = 1.0

    print("\nTesting SSL encoder...")
    with torch.no_grad():
        ssl_hidden = ssl_encoder(x1)
    print(f"  SSL hidden states shape: {ssl_hidden.shape}")

    print("\nTesting fusion model...")
    with torch.no_grad():
        output = fusion_model(ssl_hidden, x2, d)
    print(f"  Output shape: {output.shape}")
    print(f"  Output value: {output.item():.6f}")

    print("\n" + "=" * 60)
    print("Models created successfully!")
    print("=" * 60)
