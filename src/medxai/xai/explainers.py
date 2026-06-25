"""Post-hoc explainers for the trained backbone.

Grad-CAM++ (CAM-family, fast) and Integrated Gradients (gradient-family, exact)
are the two primary methods. SHAP is deferred to the faithfulness-eval stage and
run only on a small fixed subset, because image SHAP is far too slow for the full
set — generating it here would waste GPU hours.

These produce raw saliency maps. Aggregating them onto the SHARED superpixel
partition (the region-coherent protocol) happens in a later module, so the CNN
maps and the GNN attention are scored on identical regions.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from medxai.backbones.model import ResNet50MultiLabel


def load_backbone(
    ckpt_path: str, num_classes: int = 10, device: str = "cuda", dropout: float = 0.0
) -> ResNet50MultiLabel:
    """Rebuild the architecture and load trained weights. pretrained=False so it
    doesn't re-download ImageNet; our checkpoint supplies all weights."""
    model = ResNet50MultiLabel(num_classes=num_classes, pretrained=False,
                               dropout=dropout)
    ck = torch.load(ckpt_path, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    return model.to(device).eval()


def normalize_map(m: np.ndarray) -> np.ndarray:
    """Per-map min-max to [0,1] for comparable visualization/aggregation."""
    m = np.asarray(m, dtype=np.float32)
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo + 1e-8)


def gradcampp_maps(
    model: ResNet50MultiLabel,
    imgs: torch.Tensor,
    class_indices: Sequence[int],
    target_layer=None,
) -> np.ndarray:
    """Grad-CAM++ saliency for one target class per image. Target layer defaults
    to the last conv block (layer4); CAM upsamples to input resolution."""
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if target_layer is None:
        target_layer = model.backbone.layer4[-1]
    targets = [ClassifierOutputTarget(int(c)) for c in class_indices]
    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        maps = cam(input_tensor=imgs, targets=targets)  # (B, H, W) in [0,1]
    return np.stack([normalize_map(m) for m in maps])


def integrated_gradients_maps(
    model: ResNet50MultiLabel,
    imgs: torch.Tensor,
    class_indices: Sequence[int],
    n_steps: int = 32,
    baseline: torch.Tensor | None = None,
) -> np.ndarray:
    """Integrated Gradients attributions, reduced to a 2D map per image
    (abs-sum over channels). Black baseline by default."""
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)  # model(x) -> (B, num_classes) logits
    target = torch.tensor([int(c) for c in class_indices], device=imgs.device)
    base = baseline if baseline is not None else torch.zeros_like(imgs)
    attr = ig.attribute(imgs, baselines=base, target=target, n_steps=n_steps)
    maps = attr.abs().sum(dim=1).detach().cpu().numpy()  # (B, H, W)
    return np.stack([normalize_map(m) for m in maps])


def choose_target_classes(
    probs: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Per image, explain the ground-truth-present class with highest predicted
    probability (the 'true-positive present class' convention from frozen.yaml).
    Falls back to argmax prob if no GT class is present."""
    out = np.empty(len(probs), dtype=int)
    for i in range(len(probs)):
        present = np.where(labels[i] > 0)[0]
        if len(present) > 0:
            out[i] = present[np.argmax(probs[i, present])]
        else:
            out[i] = int(np.argmax(probs[i]))
    return out
