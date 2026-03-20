from __future__ import annotations

import os

from app.core.config import settings
from app.core.state import state
from app.services.bioclip2_service import (
    extract_terms_from_shard,
    load_bioclip2_model,
    load_bioclip2_text_features,
)
from app.services.retrieval import load_bioclip, load_index, load_metadata, resolve_device
from app.services.router import load_router_model

settings.ensure_yolov5_path()
from models.experimental import attempt_load  # noqa: E402


def load_all_resources() -> None:
    """
    启动时一次性加载所有模型、索引和元数据到 state。
    """
    state.runtime_device = resolve_device(settings.device)
    print(f"[Startup] Using device: {state.runtime_device}")

    # 1. Retrieval 模块
    state.model, state.preprocess = load_bioclip(state.runtime_device)
    print("[Startup] BioCLIP retrieval model loaded.")

    state.index, state.id_map = load_index()
    print("[Startup] FAISS index loaded.")

    state.id2meta = load_metadata()
    print("[Startup] Metadata loaded.")

    # 2. Router 模块
    print("[Startup] Loading router model...")
    (
        state.router_model,
        state.router_class_names,
        state.router_sonar_index,
        state.router_transform,
    ) = load_router_model(state.runtime_device)
    print(f"[Startup] Router model loaded. Classes: {state.router_class_names}")

    # 3. Sonar classifier
    print("[Startup] Loading sonar classification model...")
    state.sonar_model = attempt_load(settings.sonar_cls_path, device=state.runtime_device)
    state.sonar_model.eval()
    print(f"[Startup] Sonar model loaded. Classes: {state.sonar_model.names}")

    # 4. fish/coral 二分类
    print("[Startup] Loading fish/coral classifier...")
    state.fish_coral_model = attempt_load(settings.fish_coral_cls_path, device=state.runtime_device)
    state.fish_coral_model.eval()
    print(f"[Startup] Fish/Coral model loaded. Classes: {state.fish_coral_model.names}")

    # 5. fish detector
    print("[Startup] Loading fish detection model...")
    state.fish_model = attempt_load(settings.fish_model_path, device=state.runtime_device)
    state.fish_model.eval()
    print(f"[Startup] Fish model loaded. Classes: {state.fish_model.names}")

    # 6. coral detector
    print("[Startup] Loading coral detection model...")
    state.coral_model = attempt_load(settings.coral_model_path, device=state.runtime_device)
    state.coral_model.eval()
    print(f"[Startup] Coral model loaded. Classes: {state.coral_model.names}")

    # 7. BioCLIP2（可选）
    print("[Startup] Loading BioCLIP2 finetuned model...")
    try:
        (
            state.bioclip2_model,
            state.bioclip2_preprocess,
            state.bioclip2_tokenizer,
        ) = load_bioclip2_model(settings.bioclip2_checkpoint, state.runtime_device)

        if state.bioclip2_model is not None:
            print(f"[DEBUG] SHARD_PATH = '{settings.shard_path}'")
            print(f"[DEBUG] SHARD_PATH type = {type(settings.shard_path)}")
            print(f"[DEBUG] File exists: {os.path.exists(settings.shard_path)}")

            state.bioclip2_terms = extract_terms_from_shard(settings.shard_path, max_terms=100)

            if state.bioclip2_terms:
                (
                    state.bioclip2_text_features,
                    state.bioclip2_terms,
                ) = load_bioclip2_text_features(
                    state.bioclip2_model,
                    state.bioclip2_terms,
                    state.runtime_device,
                )
                print(f"[Startup] BioCLIP2 text features computed for {len(state.bioclip2_terms)} terms")
            else:
                print("[Startup] Warning: No terms extracted for BioCLIP2")
        else:
            print("[Startup] Warning: BIOCLIP2_MODEL is None")

    except Exception as e:
        print(f"[Startup] Warning: Failed to load BioCLIP2: {e}")
        state.bioclip2_model = None
        state.bioclip2_preprocess = None
        state.bioclip2_tokenizer = None
        state.bioclip2_text_features = None
        state.bioclip2_terms = []
