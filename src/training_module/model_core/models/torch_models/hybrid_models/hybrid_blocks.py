import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FocalLoss(nn.Module):
    def __init__(
        self, gamma: float = 2.0, alpha: float = None, size_average: bool = True, multi_class: bool = False
    ) -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.multi_class = multi_class

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.multi_class:
            ce_loss = F.cross_entropy(input, target, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        else:
            if input.size(0) != target.size(0):
                target = target.view(-1, 1)

            bce_loss = F.binary_cross_entropy_with_logits(input, target.float(), reduction="none")
            pt = torch.exp(-bce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class AttentionBlock(nn.Module):
    """Channel and spatial attention mechanism"""

    def __init__(self, in_channels: int) -> None:
        super(AttentionBlock, self).__init__()

        # Channel attention
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.spatial_attn = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        channel_attn = self.channel_pool(x)
        channel_attn = self.channel_attn(channel_attn)
        x = x * channel_attn

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.spatial_attn(spatial_out)

        return x * spatial_attn


class DenseNet121Thyroid(nn.Module):
    """
    DenseNet121-based model with attention mechanism for thyroid disease classification.
    """

    def __init__(
        self, img_channels: int = 1, num_statistical_features: int = 10, num_classes: int = 1, pretrained: int = True
    ) -> None:
        super(DenseNet121Thyroid, self).__init__()

        self.densenet = models.densenet121(weights="DEFAULT" if pretrained else None)

        if img_channels != 3:
            self.densenet.features.conv0 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.densenet_features = self.densenet.classifier.in_features

        self.densenet.classifier = nn.Identity()

        self.attention = AttentionBlock(1024)

        self.stat_branch = nn.Sequential(
            nn.Linear(num_statistical_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(int(self.densenet_features) + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, dict):
            img = x["pixels"]
            stat_features = x["features"]
        elif isinstance(x, tuple) and len(x) == 2:
            img, stat_features = x
        else:
            raise ValueError("Input must be either a dict with 'pixels' and 'features' keys or a tuple (img, features)")

        x = self.densenet.features(img)
        x = self.attention(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        stat_out = self.stat_branch(stat_features)

        combined = torch.cat((x, stat_out), dim=1)

        out = self.classifier(combined)

        return out


class ResidualMLPBlock(nn.Module):
    """MLP block with residual connection"""

    def __init__(self, in_features: int, out_features: int) -> None:
        super(ResidualMLPBlock, self).__init__()

        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)

        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        self.shortcut = None
        if in_features != out_features:
            self.shortcut = nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.linear2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = F.relu(out)

        return out


class FeatureFusionModule(nn.Module):
    """Attention-based feature fusion module"""

    def __init__(self, img_features: int, stat_features: int) -> None:
        super(FeatureFusionModule, self).__init__()

        self.img_gate = nn.Sequential(nn.Linear(img_features + stat_features, img_features), nn.Sigmoid())

        self.stat_gate = nn.Sequential(nn.Linear(img_features + stat_features, stat_features), nn.Sigmoid())

        self.fusion = nn.Linear(img_features + stat_features, img_features)

    def forward(self, img_features: torch.Tensor, stat_features: torch.Tensor) -> torch.Tensor:
        # Concatenate features
        concat_features = torch.cat([img_features, stat_features], dim=1)

        # Compute attention gates
        img_attention = self.img_gate(concat_features)
        stat_attention = self.stat_gate(concat_features)

        # Apply attention
        gated_img = img_features * img_attention
        gated_stat = stat_features * stat_attention

        # Combine gated features
        fused = torch.cat([gated_img, gated_stat], dim=1)

        # Project back to image feature dimension
        out = self.fusion(fused)

        return out


class EfficientNetThyroid(nn.Module):
    """
    EfficientNet-based model with feature fusion for thyroid disease classification.
    """

    def __init__(
        self,
        img_channels: int = 1,
        num_statistical_features: int = 10,
        num_classes: int = 1,
        model_variant: str = "b0",
        pretrained: bool = True,
    ) -> None:
        super(EfficientNetThyroid, self).__init__()

        if model_variant == "b0":
            self.effnet = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
            effnet_features = 1280
        elif model_variant == "b1":
            self.effnet = models.efficientnet_b1(weights="DEFAULT" if pretrained else None)
            effnet_features = 1280
        elif model_variant == "b2":
            self.effnet = models.efficientnet_b2(weights="DEFAULT" if pretrained else None)
            effnet_features = 1408
        elif model_variant == "b3":
            self.effnet = models.efficientnet_b3(weights="DEFAULT" if pretrained else None)
            effnet_features = 1536
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {model_variant}")

        if img_channels != 3:
            old_conv = self.effnet.features[0][0]
            new_conv = nn.Conv2d(
                img_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            self.effnet.features[0][0] = new_conv

        self.effnet.classifier = nn.Identity()

        self.mid_level_features = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        self.stat_branch = nn.Sequential(
            ResidualMLPBlock(num_statistical_features, 64), ResidualMLPBlock(64, 128), ResidualMLPBlock(128, 256)
        )

        self.fusion = FeatureFusionModule(effnet_features, 256)

        self.classifier = nn.Sequential(
            nn.Linear(effnet_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, dict):
            img = x["pixels"]
            stat_features = x["features"]
        elif isinstance(x, tuple) and len(x) == 2:
            img, stat_features = x
        else:
            raise ValueError("Input must be either a dict with 'pixels' and 'features' keys or a tuple (img, features)")

        img_features = self.effnet.features(img)
        img_features = self.effnet.avgpool(img_features)
        img_features = torch.flatten(img_features, 1)

        stat_out = self.stat_branch(stat_features)

        fused_features = self.fusion(img_features, stat_out)

        out = self.classifier(fused_features)

        return out
