"""Audio quad-stream deepfake detection model architecture."""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Stream(nn.Module):
    """A single 1-channel ResNet18 stream producing a fixed-size embedding."""

    def __init__(self, input_channels: int = 1, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()
        if input_channels != 1:
            raise ValueError(f"Expected input_channels=1 (got {input_channels})")

        backbone = models.resnet18(pretrained=pretrained)

        # Replace first conv to accept 1 channel instead of 3.
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            input_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # If pretrained, adapt RGB conv1 weights -> single channel by averaging across RGB.
        if pretrained and hasattr(old_conv, "weight") and old_conv.weight is not None:
            with torch.no_grad():
                backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Remove final classification head; keep global average pooling output.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B, 512, 1, 1)
        self.fc = nn.Linear(512, feature_dim)
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out", nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

        # If true, the backbone runs frozen (no grads, eval mode) but the per-stream FC can still train.
        self.backbone_frozen = False

    def freeze_backbone(self, freeze: bool = True, keep_eval: bool = True) -> None:
        """Freeze/unfreeze the ResNet backbone (not the per-stream FC).

        When frozen, we:
          - set backbone params requires_grad=False
          - run backbone forward under torch.no_grad() (saves memory)
          - keep backbone in eval() so BN/Dropout stay stable
        """
        self.backbone_frozen = bool(freeze)
        for p in self.backbone.parameters():
            p.requires_grad = not self.backbone_frozen
        if self.backbone_frozen and keep_eval:
            self.backbone.eval()

    def train(self, mode: bool = True):
        # Keep normal train/eval behavior for the full module, but force frozen backbone to eval().
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_frozen:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class QuadStreamModel(nn.Module):
    """Audio quad-stream model: segment/full × STFT/Log-Mel (4 streams)."""

    def __init__(
        self,
        backbone: str = "resnet18",
        feature_dim: int = 256,
        fusion_dim: int = 512,
        dropout: float = 0.5,
        pretrained: bool = True,
        use_attention: bool = False,
        freeze_stft_backbones: bool = False,
        freeze_logmel_backbones: bool = False,
    ):
        super().__init__()

        if str(backbone).lower() != "resnet18":
            raise ValueError(f"Only resnet18 is supported for all streams (got backbone={backbone!r})")

        self.use_attention = bool(use_attention)

        # Four 1-channel ResNet18 streams (B,1,224,224) -> (B,feature_dim)
        self.segment_stft_stream = ResNet18Stream(input_channels=1, feature_dim=feature_dim, pretrained=pretrained)
        self.segment_logmel_stream = ResNet18Stream(input_channels=1, feature_dim=feature_dim, pretrained=pretrained)
        self.full_stft_stream = ResNet18Stream(input_channels=1, feature_dim=feature_dim, pretrained=pretrained)
        self.full_logmel_stream = ResNet18Stream(input_channels=1, feature_dim=feature_dim, pretrained=pretrained)

        # Optionally freeze ONLY the STFT backbones (used at forward() lines ~130 and ~132).
        if bool(freeze_stft_backbones):
            self.freeze_stft_backbones()

        # Optionally freeze ONLY the Log-Mel backbones.
        if bool(freeze_logmel_backbones):
            self.freeze_logmel_backbones()

        if self.use_attention:
            self.attention_weights = nn.Linear(feature_dim * 4, 4)

        # Fusion layers - fusing 4 streams
        self.fusion = nn.Sequential(
            # Regularize the concatenated 4-stream embedding before any mixing/projection.
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize fusion and classifier layers
        self._initialize_fusion_layers()

    def freeze_all_backbones(self) -> None:
        """Freeze backbones for all 4 streams (keeps per-stream FC + fusion + classifier trainable)."""
        for stream in (
            self.segment_stft_stream,
            self.segment_logmel_stream,
            self.full_stft_stream,
            self.full_logmel_stream,
        ):
            stream.freeze_backbone(True)

    def freeze_stft_backbones(self) -> None:
        """Freeze backbones for the two STFT streams only."""
        self.segment_stft_stream.freeze_backbone(True)
        self.full_stft_stream.freeze_backbone(True)

    def freeze_logmel_backbones(self) -> None:
        """Freeze backbones for the two Log-Mel streams only."""
        self.segment_logmel_stream.freeze_backbone(True)
        self.full_logmel_stream.freeze_backbone(True)
    
    def _initialize_fusion_layers(self):
        """Initialize fusion and classifier layers with proper weights."""
        for module in self.fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Initialize classifier with smaller weights to start near 0.5 (random guess)
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)  # Start near 0.5 after sigmoid
    
    def forward(
        self,
        segment_stft: torch.Tensor,
        segment_logmel: torch.Tensor,
        full_stft: torch.Tensor,
        full_logmel: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            segment_stft: (B, 1, 224, 224)
            segment_logmel: (B, 1, 224, 224)
            full_stft: (B, 1, 224, 224)
            full_logmel: (B, 1, 224, 224)

        Returns:
            Probability (B, 1) after sigmoid.
        """
        # Extract embeddings from all 4 streams
        seg_stft_f = self.segment_stft_stream(segment_stft)
        seg_mel_f = self.segment_logmel_stream(segment_logmel)
        full_stft_f = self.full_stft_stream(full_stft)
        full_mel_f = self.full_logmel_stream(full_logmel)

        # Fusion
        if self.use_attention:
            concat_features = torch.cat([seg_stft_f, seg_mel_f, full_stft_f, full_mel_f], dim=1)
            attention_logits = self.attention_weights(concat_features)
            attention_weights = torch.softmax(attention_logits, dim=1)

            seg_stft_f = seg_stft_f * attention_weights[:, 0:1]
            seg_mel_f = seg_mel_f * attention_weights[:, 1:2]
            full_stft_f = full_stft_f * attention_weights[:, 2:3]
            full_mel_f = full_mel_f * attention_weights[:, 3:4]

        fused = torch.cat([seg_stft_f, seg_mel_f, full_stft_f, full_mel_f], dim=1)
        
        # Pass through fusion layers
        fused = self.fusion(fused)
        
        # Classification
        output = self.classifier(fused)
        
        return output


