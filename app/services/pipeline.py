from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from PIL import Image

from app.core.config import settings
from app.core.state import state
from app.services.bioclip2_service import predict_with_bioclip2
from app.services.classifiers import predict_with_yolo_classifier, predict_with_yolo_detector
from app.services.retrieval import run_faiss_query
from app.services.router import run_router_classification


def generate_request_id() -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"{now}_{short_id}"


def infer_image_type_from_meta(meta: Dict[str, Any]) -> str:
    domain = meta.get("domain")
    if domain in ("sonar", "biological"):
        return domain
    return "unknown"


def build_default_modules() -> Dict[str, Any]:
    return {
        "retrieval": None,
        "router": None,
        "sonar": None,
        "fish_coral_classifier": None,
        "fish": None,
        "coral": None,
        "bioclip2": None,
        "fusion": None,
    }


def build_final_result(
    status: str,
    source: Optional[str],
    image_type: Optional[str],
    primary_label: Optional[str],
    all_labels: Optional[List[str]],
    confidence: Optional[float],
    display_text: str,
    note: str,
) -> Dict[str, Any]:
    return {
        "status": status,
        "source": source,
        "image_type": image_type,
        "primary_label": primary_label,
        "all_labels": all_labels or [],
        "confidence": confidence,
        "display_text": display_text,
        "note": note,
    }

def normalize_compare_label(label: Optional[str]) -> str:
    if not label:
        return ""
    return str(label).strip().lower().replace(" ", "")


def extract_name_from_bioclip_term(term: Optional[str]) -> str:
    """
    BioCLIP term 可能是：
    1) 纯名字: "Colpophyllia Natans"
    2) 生物链: "Animalia > ... > Colpophyllia natans"

    这里统一提取“最后一个名字”用于比较；
    如果本身就是纯名字，则直接返回原值。
    """
    if not term:
        return ""

    s = str(term).strip()
    if not s:
        return ""

    if ">" in s:
        parts = [p.strip() for p in s.split(">") if p.strip()]
        if parts:
            return parts[-1]

    return s


def get_bioclip_topk_matches(
    bioclip2_result: Optional[Dict[str, Any]],
    topk: int = 5,
) -> List[Dict[str, Any]]:
    if not bioclip2_result:
        return []

    matches = bioclip2_result.get("matches") or []
    if not isinstance(matches, list):
        return []

    return matches[:topk]


def yolo_label_in_bioclip_topk(
    yolo_label: Optional[str],
    bioclip2_result: Optional[Dict[str, Any]],
    topk: int = 5,
) -> bool:
    yolo_norm = normalize_compare_label(yolo_label)
    if not yolo_norm:
        return False

    top_matches = get_bioclip_topk_matches(bioclip2_result, topk=topk)

    for m in top_matches:
        if not isinstance(m, dict):
            continue

        term = m.get("term")
        bioclip_name = extract_name_from_bioclip_term(term)
        bioclip_norm = normalize_compare_label(bioclip_name)

        if yolo_norm == bioclip_norm:
            return True

    return False

def fuse_biological_results(
    fish_coral_result: Dict[str, Any],
    yolo_result: Optional[Dict[str, Any]],
    bioclip2_result: Optional[Dict[str, Any]],
    bioclip_match_topk: int = 5,
) -> Dict[str, Any]:
    coarse_type = fish_coral_result["primary_label"].lower() if fish_coral_result else None

    selected_source = None
    primary_label = None
    confidence = None
    all_labels = []
    note_parts = []

    yolo_primary_label = None
    yolo_confidence = None
    yolo_model_name = None

    if yolo_result is not None:
        yolo_model_name = yolo_result.get("model_name")
        yolo_primary_label = yolo_result.get("primary_label")
        yolo_confidence = yolo_result.get("confidence")

    bioclip_primary_term = None
    bioclip_primary_name = None
    bioclip_confidence = None
    bioclip_terms = []

    if bioclip2_result is not None:
        if bioclip2_result.get("matches"):
            bioclip_terms = [
                m["term"] for m in bioclip2_result["matches"]
                if isinstance(m, dict) and "term" in m
            ]

        primary_match = bioclip2_result.get("primary_match")
        if isinstance(primary_match, dict):
            bioclip_primary_term = primary_match.get("term")
            bioclip_primary_name = extract_name_from_bioclip_term(bioclip_primary_term)

        bioclip_confidence = bioclip2_result.get("confidence")

    # all_labels 保留原始展示内容：先 YOLO，再补 BioCLIP 原始 term
    if yolo_result is not None:
        all_labels.extend(yolo_result.get("all_labels", []))

    for t in bioclip_terms:
        if t not in all_labels:
            all_labels.append(t)

    # 核心融合逻辑
    # 1) YOLO 和 BioCLIP 都有：
    #    - 如果 YOLO 标签出现在 BioCLIP topK（提取名字 + 归一化比较），用 YOLO
    #    - 否则用 BioCLIP
    # 2) 只有 YOLO：用 YOLO
    # 3) 只有 BioCLIP：用 BioCLIP
    # 4) 如果 BioCLIP 被选中且 primary_match 是信息链，则最终只取最后一个名字展示
    if yolo_primary_label is not None and bioclip2_result is not None and bioclip_terms:
        matched = yolo_label_in_bioclip_topk(
            yolo_label=yolo_primary_label,
            bioclip2_result=bioclip2_result,
            topk=bioclip_match_topk,
        )

        if matched:
            selected_source = yolo_model_name
            primary_label = yolo_primary_label
            confidence = yolo_confidence
            note_parts.append(
                f"YOLO label matched BioCLIP2 top{bioclip_match_topk} after normalized-name comparison; YOLO result selected."
            )
        else:
            selected_source = "bioclip2"
            primary_label = bioclip_primary_name
            confidence = bioclip_confidence
            note_parts.append(
                f"YOLO label not found in BioCLIP2 top{bioclip_match_topk} after normalized-name comparison; BioCLIP2 result selected."
            )

        note_parts.append("BioCLIP2 terms added as biological-chain support.")

    elif yolo_primary_label is not None:
        selected_source = yolo_model_name
        primary_label = yolo_primary_label
        confidence = yolo_confidence
        note_parts.append("BioCLIP2 unavailable, YOLO result selected.")

    elif bioclip_primary_name is not None:
        selected_source = "bioclip2"
        primary_label = bioclip_primary_name
        confidence = bioclip_confidence
        note_parts.append("YOLO unavailable, fallback to BioCLIP2 primary match.")

    else:
        note_parts.append("Neither YOLO nor BioCLIP2 produced a valid species prediction.")

    if coarse_type == "fish":
        image_type = "fish"
    elif coarse_type == "coral":
        image_type = "coral"
    else:
        image_type = "biological_unknown"

    display_text = f"生物识别结果：{primary_label}" if primary_label else "生物识别结果：未确定"

    return {
        "enabled": True,
        "coarse_type": coarse_type,
        "selected_source": selected_source,
        "primary_label": primary_label,
        "confidence": confidence,
        "all_labels": all_labels,
        "reason": " ".join(note_parts),
        "display_text": display_text,
        "image_type": image_type,
    }


def run_full_pipeline(image_path: Path) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")

    faiss_result = run_faiss_query(image_path)
    retrieval_module = faiss_result["retrieval_module"]
    top1_meta = faiss_result["top1_meta"]

    modules = build_default_modules()
    modules["retrieval"] = retrieval_module

    if retrieval_module["db_hit"]:
        labels = retrieval_module["top1_labels"]
        primary_label = labels[0] if labels else None
        image_type = infer_image_type_from_meta(top1_meta)
        confidence = retrieval_module["top1_similarity"]

        final_result = build_final_result(
            status="success",
            source="database",
            image_type=image_type,
            primary_label=primary_label,
            all_labels=labels,
            confidence=confidence,
            display_text=f"识别结果：{primary_label}",
            note="Matched from retrieval database.",
        )

        return {
            "success": True,
            "stage": "db_hit",
            "final_result": final_result,
            "modules": modules,
        }

    # ===== 改动1：bioclip2 提前到 router 前，无条件执行 =====
    bioclip2_result = None
    if (
        settings.use_bioclip2
        and state.bioclip2_model is not None
        and state.bioclip2_text_features is not None
    ):
        bioclip2_result = predict_with_bioclip2(
            image=image,
            model=state.bioclip2_model,
            preprocess=state.bioclip2_preprocess,
            text_features=state.bioclip2_text_features,
            terms=state.bioclip2_terms,
            device=state.runtime_device,
            topk=5,
        )
        modules["bioclip2"] = bioclip2_result

    router_result = run_router_classification(image_path)
    modules["router"] = router_result["router_module"]

    image_type = router_result["predicted_type"]
    stage = router_result["stage"]

    if image_type == "sonar":
        result = predict_with_yolo_classifier(
            state.sonar_model,
            image,
            state.runtime_device,
            "MergedData_7",
        )
        modules["sonar"] = result

        # 可选：把 bioclip2 是否执行写进 note
        note = "Sonar classification completed."
        if bioclip2_result is not None:
            note += " BioCLIP2 was also executed unconditionally."

        final_result = build_final_result(
            status="success",
            source="sonar_model",
            image_type="sonar",
            primary_label=result["primary_label"],
            all_labels=result["all_labels"],
            confidence=result["confidence"],
            display_text=f"声纳图像识别：{result['primary_label']}",
            note=note,
        )
    else:
        fish_coral_result = predict_with_yolo_classifier(
            state.fish_coral_model,
            image,
            state.runtime_device,
            "fish_coral_cls",
        )
        modules["fish_coral_classifier"] = fish_coral_result

        coarse_label = fish_coral_result["primary_label"].lower()
        yolo_species_result = None

        if coarse_label == "fish":
            yolo_species_result = predict_with_yolo_detector(
                state.fish_model,
                image,
                state.runtime_device,
                "merge_fish_small",
            )
            modules["fish"] = yolo_species_result
        else:
            yolo_species_result = predict_with_yolo_detector(
                state.coral_model,
                image,
                state.runtime_device,
                "Coral_one2",
            )
            modules["coral"] = yolo_species_result

        # ===== 改动2：这里不再重复跑 bioclip2，直接复用上面结果 =====
        fusion_result = fuse_biological_results(
            fish_coral_result=fish_coral_result,
            yolo_result=yolo_species_result,
            bioclip2_result=bioclip2_result,
        )
        modules["fusion"] = fusion_result

        final_result = build_final_result(
            status="success",
            source=fusion_result["selected_source"],
            image_type=fusion_result["image_type"],
            primary_label=fusion_result["primary_label"],
            all_labels=fusion_result["all_labels"],
            confidence=fusion_result["confidence"],
            display_text=fusion_result["display_text"],
            note=fusion_result["reason"],
        )

        stage = "bio_fused"

    return {
        "success": True,
        "stage": stage,
        "final_result": final_result,
        "modules": modules,
    }
