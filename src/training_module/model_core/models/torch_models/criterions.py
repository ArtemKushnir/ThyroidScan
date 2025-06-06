from pytorch_toolbelt.losses import BinaryFocalLoss, DiceLoss, FocalLoss, SoftBCEWithLogitsLoss, SoftCrossEntropyLoss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

BINARY_CRITERION = {
    "bce_with_logits": BCEWithLogitsLoss,
    "soft_bce_with_logits": SoftBCEWithLogitsLoss,
    "focal": BinaryFocalLoss,
    "dice": DiceLoss,
    "cross_entropy": CrossEntropyLoss,
}

MULTICLASS_CRITERION = {
    "focal": FocalLoss,
    "dice": DiceLoss,
    "cross_entropy": CrossEntropyLoss,
    "soft_cross_entropy": SoftCrossEntropyLoss,
}
