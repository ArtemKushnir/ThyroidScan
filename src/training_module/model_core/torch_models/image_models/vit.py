# from typing import Any, Callable, Optional, Union
#
# from monai.networks.nets import ViT
# from monai.transforms import NormalizeIntensity, Resize, ScaleIntensity
# from torch import nn
#
# from src.training_module.model_core.base_models import PyTorchModel
#
#
# class ThyroidViTModel(PyTorchModel):
#     """
#     Vision Transformer (ViT) implementation for thyroid ultrasound classification.
#     Transformers have shown promising results in medical imaging by capturing global context
#     and relationships between different regions of the image.
#     """
#
#     def __init__(
#         self,
#         optimizer: str = "adam",
#         criterion: str = "bce_logit",
#         model_params: Optional[dict[str, Any]] = None,
#         is_binary: bool = True,
#         pretrained: bool = True,
#         spatial_dims: int = 2,
#         n_input_channels: int = 1,
#         num_classes: int = 1,
#         img_size: Union[int, tuple[int, int]] = (224, 224),
#         patch_size: int = 16,
#         hidden_size: int = 768,
#         mlp_dim: int = 3072,
#         num_layers: int = 12,
#         num_heads: int = 12,
#         pos_embed: str = "conv",
#         classification: bool = True,
#     ) -> None:
#         super().__init__(optimizer, criterion, model_params, is_binary)
#
#         self.pretrained = pretrained
#         self.spatial_dims = spatial_dims
#         self.n_input_channels = n_input_channels
#         self.num_classes = num_classes
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.hidden_size = hidden_size
#         self.mlp_dim = mlp_dim
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.pos_embed = pos_embed
#         self.classification = classification
#
#         self.model = self._create_model()
#         self.preprocessing = self._initialize_preprocessing()
#
#     def _create_model(self) -> nn.Module:
#         out_classes = 1 if self.is_binary else self.num_classes
#
#         model = ViT(
#             in_channels=self.n_input_channels,
#             img_size=self.img_size,
#             patch_size=self.patch_size,
#             hidden_size=self.hidden_size,
#             mlp_dim=self.mlp_dim,
#             num_layers=self.num_layers,
#             num_heads=self.num_heads,
#             pos_embed=self.pos_embed,
#             classification=self.classification,
#             num_classes=out_classes,
#             spatial_dims=self.spatial_dims,
#             pretrained=self.pretrained,
#         )
#
#         return model
#
#     def _initialize_preprocessing(self) -> list[Callable]:
#         input_size = self.img_size if isinstance(self.img_size, tuple) else (self.img_size, self.img_size)
#
#         transforms = [
#             lambda x: x.unsqueeze(1) if x.dim() == 3 else x,
#             Resize(spatial_size=input_size),
#             ScaleIntensity(),
#             NormalizeIntensity(subtrahend=0.5, divisor=0.5),  # Standard ViT normalization
#         ]
#
#         return transforms
#
#     def _get_additional_state(self) -> dict[str, Any]:
#         return {
#             "pretrained": self.pretrained,
#             "spatial_dims": self.spatial_dims,
#             "n_input_channels": self.n_input_channels,
#             "num_classes": self.num_classes,
#             "img_size": self.img_size,
#             "patch_size": self.patch_size,
#             "hidden_size": self.hidden_size,
#             "mlp_dim": self.mlp_dim,
#             "num_layers": self.num_layers,
#             "num_heads": self.num_heads,
#             "pos_embed": self.pos_embed,
#             "classification": self.classification,
#         }
#
#     def _load_additional_state(self, checkpoint: dict[str, Any]) -> None:
#         self.pretrained = checkpoint.get("pretrained", True)
#         self.spatial_dims = checkpoint.get("spatial_dims", 2)
#         self.n_input_channels = checkpoint.get("n_input_channels", 1)
#         self.num_classes = checkpoint.get("num_classes", 1)
#         self.img_size = checkpoint.get("img_size", (224, 224))
#         self.patch_size = checkpoint.get("patch_size", 16)
#         self.hidden_size = checkpoint.get("hidden_size", 768)
#         self.mlp_dim = checkpoint.get("mlp_dim", 3072)
#         self.num_layers = checkpoint.get("num_layers", 12)
#         self.num_heads = checkpoint.get("num_heads", 12)
#         self.pos_embed = checkpoint.get("pos_embed", "conv")
#         self.classification = checkpoint.get("classification", True)
#
#         if self.model is None:
#             self.model = self._create_model()
#             self.preprocessing = self._initialize_preprocessing()
