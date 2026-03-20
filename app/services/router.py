from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from app.core.config import settings
from app.core.state import state

settings.ensure_yolov5_path()

from models.experimental import attempt_load  # noqa: E402
from utils.augmentations import classify_transforms  # noqa: E402


def _find_sonar_index(names_map: dict) -> int:
    for idx, name in names_map.items():
        if str(name).strip().lower() == "sonar":
            return int(idx)
    return 0


def load_router_model(device: str):
    model = attempt_load(settings.router_model_path, device=device)
    model.eval()
    class_names = model.names
    sonar_index = _find_sonar_index(class_names)
    transform = classify_transforms(224)
    return model, class_names, sonar_index, transform


@torch.no_grad()
def run_router_classification(image_path: Path) -> Dict[str, Any]:
    if state.router_model is None or state.router_transform is None or state.runtime_device is None:
        raise RuntimeError("Router model not initialized correctly")

    img_pil = Image.open(image_path).convert("RGB")
    im = np.array(img_pil)[:, :, ::-1].copy()
    img_tensor = state.router_transform(im).unsqueeze(0).to(state.runtime_device)

    out = state.router_model(img_tensor)
    logits = out[0] if isinstance(out, (list, tuple)) else out
    probs = torch.softmax(logits, dim=1).squeeze(0)

    top1_prob, top1_idx = torch.max(probs, dim=0)
    top1_prob = float(top1_prob.item())
    top1_idx = int(top1_idx.item())

    sonar_prob = float(probs[state.router_sonar_index].item())
    is_sonar = sonar_prob >= settings.router_threshold

    top1_label_raw = str(state.router_class_names.get(top1_idx, str(top1_idx)))

    predicted_type = "sonar" if is_sonar else "biological"
    stage = "sonar_routed" if is_sonar else "bio_routed"

    router_module = {
        "enabled": True,
        "predicted_type": predicted_type,
        "confidence": round(sonar_prob if is_sonar else (1.0 - sonar_prob), 4),
        "model_name": Path(settings.router_model_path).name,
        "raw_top1_label": top1_label_raw,
        "raw_top1_confidence": round(top1_prob, 4),
        "sonar_probability": round(sonar_prob, 4),
        "threshold": settings.router_threshold,
    }

    return {
        "stage": stage,
        "predicted_type": predicted_type,
        "confidence": router_module["confidence"],
        "router_module": router_module,
    }
