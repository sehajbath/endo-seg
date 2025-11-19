"""Swin UNETR wrapper with MC-dropout and TTA uncertainty utilities."""

from __future__ import annotations

from functools import partial
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete

Tensor = torch.Tensor


class SwinUNETRWithUncertainty(nn.Module):
    """Wrap MONAI's SwinUNETR with helper utilities for uncertainty estimation."""

    def __init__(
        self,
        img_size: Sequence[int],
        in_channels: int,
        out_channels: int,
        feature_size: int = 48,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = True,
        roi_size: Sequence[int] = (128, 128, 32),
        sw_batch_size: int = 2,
        infer_overlap: float = 0.5,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.backbone = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
        )

        self.roi_size = tuple(roi_size)
        self.sw_batch_size = sw_batch_size
        self.infer_overlap = infer_overlap

        self.model_inferer = partial(
            sliding_window_inference,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.backbone,
            overlap=self.infer_overlap,
        )

        self.post_pred = AsDiscrete(argmax=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def infer_sliding_window(self, x: Tensor) -> Tensor:
        """Deterministic sliding-window inference used for validation."""
        return self.model_inferer(x)

    def enable_mc_dropout(self) -> None:
        """Enable dropout layers while keeping the rest of the model in eval mode."""
        self.backbone.eval()
        for module in self.backbone.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                module.train()

    def disable_mc_dropout(self) -> None:
        self.backbone.eval()

    def probs_to_labels(self, prob_map: Tensor) -> Tensor:
        return self.post_pred(prob_map)

    @staticmethod
    def tensor_to_numpy(tensor: Tensor) -> "numpy.ndarray":  # type: ignore[name-defined]
        return tensor.detach().cpu().numpy()

    def _default_tta_transforms(self) -> List[Callable[[Tensor], Tuple[Tensor, Callable[[Tensor], Tensor]]]]:
        transforms = []

        def identity(x: Tensor) -> Tuple[Tensor, Callable[[Tensor], Tensor]]:
            return x, lambda y: y

        transforms.append(identity)

        for dim in (2, 3, 4):
            transforms.append(self._make_flip_transform(dim))

        return transforms

    @staticmethod
    def _make_flip_transform(dim: int) -> Callable[[Tensor], Tuple[Tensor, Callable[[Tensor], Tensor]]]:
        def _transform(x: Tensor) -> Tuple[Tensor, Callable[[Tensor], Tensor]]:
            flipped = torch.flip(x, dims=(dim,))
            return flipped, lambda y: torch.flip(y, dims=(dim,))

        return _transform

    def predict_with_uncertainty(
        self,
        x: Tensor,
        num_mc_samples: int = 8,
        num_tta: int = 4,
        tta_transforms: Optional[Iterable[Callable[[Tensor], Tuple[Tensor, Callable[[Tensor], Tensor]]]]] = None,
        process_logits: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run MC-dropout + TTA inference and return predictive statistics."""

        transforms = list(tta_transforms or self._default_tta_transforms())
        if num_tta > 0:
            transforms = transforms[: min(num_tta, len(transforms))]
        else:
            transforms = transforms[:1]

        prob_samples: List[Tensor] = []
        total_mc = max(1, num_mc_samples)

        with torch.no_grad():
            for _ in range(total_mc):
                if num_mc_samples > 1:
                    self.enable_mc_dropout()
                else:
                    self.disable_mc_dropout()

                tta_probs: List[Tensor] = []
                for transform in transforms:
                    aug_x, inverse_fn = transform(x)
                    logits = self.model_inferer(aug_x.to(self.device))
                    logits = inverse_fn(logits)
                    probs = torch.softmax(logits, dim=1) if process_logits else logits
                    tta_probs.append(probs)

                prob_samples.append(torch.stack(tta_probs, dim=0))

        self.disable_mc_dropout()

        prob_tensor = torch.stack(prob_samples, dim=0)  # [mc, tta, B, C, H, W, D]
        mean_probs = prob_tensor.mean(dim=(0, 1))

        # Predictive entropy
        entropy = -torch.sum(mean_probs * torch.log(mean_probs.clamp(min=1e-8)), dim=1, keepdim=True)
        entropy = entropy / torch.log(torch.tensor(mean_probs.shape[1], device=mean_probs.device, dtype=mean_probs.dtype))

        tta_mean = prob_tensor.mean(dim=1)
        epistemic_var = tta_mean.var(dim=0, unbiased=False)
        aleatoric_var = prob_tensor.var(dim=1, unbiased=False).mean(dim=0)

        return mean_probs, entropy, epistemic_var, aleatoric_var


__all__ = ["SwinUNETRWithUncertainty"]
