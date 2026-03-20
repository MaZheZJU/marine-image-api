from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


@dataclass
class Settings:
    app_title: str = os.getenv("APP_TITLE", "Marine Image Retrieval API")
    app_version: str = os.getenv("APP_VERSION", "0.4.0")

    device: str = os.getenv("DEVICE", "cuda")
    topk: int = _get_int("TOPK", 5)
    threshold: float = _get_float("THRESHOLD", 0.90)
    router_threshold: float = _get_float("ROUTER_THRESHOLD", 0.5)
    use_bioclip2: bool = _get_bool("USE_BIOCLIP2", True)

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _get_int("PORT", 8000)

    yolov5_dir: str = os.getenv("YOLOV5_DIR", "/home/mazhe/yolov5")
    model_dir: str = os.getenv("MODEL_DIR", "/data/mazhe/models/bioclip")
    metadata: str = os.getenv("METADATA", "/data/mazhe/origin_image/MergedData/metadata.jsonl")
    index_dir: str = os.getenv("INDEX_DIR", "/data/mazhe/origin_image/MergedData/faiss_sonar")

    router_model_path: str = os.getenv(
        "ROUTER_MODEL_PATH",
        "/home/mazhe/yolo_pt/sonar_other_classification/weights/best.pt",
    )
    sonar_cls_path: str = os.getenv(
        "SONAR_CLS_PATH",
        "/home/mazhe/yolo_pt/MergedData_7/weights/best.pt",
    )
    fish_coral_cls_path: str = os.getenv(
        "FISH_CORAL_CLS_PATH",
        "/home/mazhe/yolo_pt/fish_coral_cls/weights/best.pt",
    )
    fish_model_path: str = os.getenv(
        "FISH_MODEL_PATH",
        "/home/mazhe/yolo_pt/merge_fish_small/weights/best.pt",
    )
    coral_model_path: str = os.getenv(
        "CORAL_MODEL_PATH",
        "/home/mazhe/yolo_pt/Coral_one2/weights/best.pt",
    )
    bioclip2_checkpoint: str = os.getenv(
        "BIOCLIP2_CHECKPOINT",
        "/data/mazhe/training_logs/taxon_v2/taxon_v211/checkpoints/epoch_50.pt",
    )
    shard_path: str = os.getenv(
        "SHARD_PATH",
        "/data/mazhe/webdataset_fixed/shard-000000.tar",
    )

    def ensure_yolov5_path(self) -> None:
        yolov5_dir = str(Path(self.yolov5_dir))
        if yolov5_dir not in sys.path:
            sys.path.append(yolov5_dir)

    def as_dict(self) -> dict:
        return {
            "app_title": self.app_title,
            "app_version": self.app_version,
            "device": self.device,
            "topk": self.topk,
            "threshold": self.threshold,
            "router_threshold": self.router_threshold,
            "use_bioclip2": self.use_bioclip2,
            "host": self.host,
            "port": self.port,
            "yolov5_dir": self.yolov5_dir,
            "model_dir": self.model_dir,
            "metadata": self.metadata,
            "index_dir": self.index_dir,
            "router_model_path": self.router_model_path,
            "sonar_cls_path": self.sonar_cls_path,
            "fish_coral_cls_path": self.fish_coral_cls_path,
            "fish_model_path": self.fish_model_path,
            "coral_model_path": self.coral_model_path,
            "bioclip2_checkpoint": self.bioclip2_checkpoint,
            "shard_path": self.shard_path,
        }


settings = Settings()


def print_settings_summary() -> None:
    print("[Config] Current settings:")
    for k, v in settings.as_dict().items():
        print(f"  - {k}: {v}")
