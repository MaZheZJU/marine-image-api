#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


sys.path.append('/data/mazhe/yolov5')
from models.experimental import attempt_load
import faiss
import numpy as np
from PIL import Image
import torch
import open_clip
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# =========================
# YOLOv5 路径（router 分类模型依赖）
# =========================
YOLOV5_DIR = "/home/mazhe/yolov5"
if YOLOV5_DIR not in sys.path:
    sys.path.append(YOLOV5_DIR)

from models.experimental import attempt_load
from utils.augmentations import classify_transforms


# =========================
# 配置区：按你的实际路径修改
# =========================
MODEL_DIR = "/data/mazhe/models/bioclip"
METADATA = "/data/mazhe/origin_image/MergedData/metadata.jsonl"
INDEX_DIR = "/data/mazhe/origin_image/MergedData/faiss_sonar"

# router 模型：sonar / biological
ROUTER_MODEL_PATH = "/home/mazhe/yolo_pt/sonar_other_classification/weights/best.pt"
ROUTER_THRESHOLD = 0.5

# 新增模型路径
SONAR_CLS_PATH = "/home/mazhe/yolo_pt/MergedData_7/weights/best.pt"  # 声纳图识别模型
FISH_CORAL_CLS_PATH = "/home/mazhe/yolo_pt/fish_coral_cls/weights/best.pt"  # fish/coral 二分类
FISH_MODEL_PATH = "/home/mazhe/yolo_pt/merge_fish_small/weights/best.pt"  # 鱼类分类模型
CORAL_MODEL_PATH = "/home/mazhe/yolo_pt/Coral_one2/weights/best.pt"  # 珊瑚分类模型
BIOCLIP2_CHECKPOINT = "/data/mazhe/training_logs/taxon_v2/taxon_v211/checkpoints/epoch_50.pt"  # BioCLIP2微调模型
SHARD_PATH = "/data/mazhe/webdataset_fixed/shard-000000.tar"  # 用于提取术语的shard

# 是否使用 BioCLIP2（True=用BioCLIP2，False=用YOLO）
USE_BIOCLIP2 = True

TOPK = 5
THRESHOLD = 0.90
DEVICE = "cuda"

APP_TITLE = "Marine Image Retrieval API"
APP_VERSION = "0.4.0"  # 升级版本号


# =========================
# 全局缓存：启动时加载一次
# =========================
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

MODEL = None
PREPROCESS = None
INDEX = None
ID_MAP = None
ID2META = None
RUNTIME_DEVICE = None

ROUTER_MODEL = None
ROUTER_CLASS_NAMES = {}
ROUTER_SONAR_INDEX = 0
ROUTER_TRANSFORM = None

# 新增全局变量
SONAR_MODEL = None
FISH_CORAL_MODEL = None
FISH_MODEL = None
CORAL_MODEL = None
BIOCLIP2_MODEL = None
BIOCLIP2_PREPROCESS = None
BIOCLIP2_TOKENIZER = None
BIOCLIP2_TEXT_FEATURES = None
BIOCLIP2_TERMS = []


# =========================
# 工具函数
# =========================
def resolve_device(preferred_device: str) -> str:
    if preferred_device.startswith("cuda") and torch.cuda.is_available():
        return preferred_device
    return "cpu"


def prepare_image_for_classification(image_path: Path, device: str) -> torch.Tensor:
    """
    统一的图像预处理函数，用于所有分类模型
    基于 YOLOv5 的 classify_transforms
    """
    from utils.augmentations import classify_transforms
    
    # 读取图像
    img_pil = Image.open(image_path).convert("RGB")
    
    # 转换为 numpy (RGB -> BGR) 因为 classify_transforms 期望 BGR
    im = np.array(img_pil)[:, :, ::-1].copy()
    
    # 应用分类预处理
    img_tensor = classify_transforms(224)(im).unsqueeze(0).to(device)
    
    return img_tensor


def process_classification_result(model_result, model_names: dict) -> Dict[str, Any]:
    """
    处理分类模型的结果
    """
    # 获取预测结果
    pred = model_result[0].pred[0] if hasattr(model_result[0], 'pred') else model_result[0]
    probs = torch.softmax(pred, dim=0)
    top1_idx = probs.argmax().item()
    top1_conf = probs[top1_idx].item()
    
    # 获取所有类别的概率
    all_probs = probs.cpu().numpy()
    all_labels = [model_names[i] for i in range(len(all_probs))]
    
    return {
        "primary_label": model_names[top1_idx],
        "confidence": top1_conf,
        "all_labels": all_labels,
        "all_probabilities": all_probs.tolist()
    }


def generate_request_id() -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"{now}_{short_id}"


def load_bioclip(device: str):
    out = open_clip.create_model_from_pretrained(f"local-dir:{MODEL_DIR}")

    if isinstance(out, (tuple, list)) and len(out) == 2:
        model, preprocess = out
    elif isinstance(out, (tuple, list)) and len(out) == 3:
        model, preprocess, _ = out
    else:
        raise RuntimeError("Unexpected return from create_model_from_pretrained")

    model = model.to(device).eval()
    return model, preprocess


def load_index():
    index_path = Path(INDEX_DIR) / "index.faiss"
    id_map_path = Path(INDEX_DIR) / "id_map.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not id_map_path.exists():
        raise FileNotFoundError(f"id_map not found: {id_map_path}")

    index = faiss.read_index(str(index_path))
    with id_map_path.open("r", encoding="utf-8") as f:
        id_map = json.load(f)

    return index, id_map


def load_metadata():
    metadata_path = Path(METADATA)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    id2meta = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "id" in rec:
                id2meta[rec["id"]] = rec
    return id2meta


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
        "bioclip2": None
    }


def build_final_result(
    status: str,
    source: Optional[str],
    image_type: Optional[str],
    primary_label: Optional[str],
    all_labels: Optional[List[str]],
    confidence: Optional[float],
    display_text: str,
    note: str
) -> Dict[str, Any]:
    return {
        "status": status,
        "source": source,
        "image_type": image_type,
        "primary_label": primary_label,
        "all_labels": all_labels or [],
        "confidence": confidence,
        "display_text": display_text,
        "note": note
    }


# =========================
# FAISS / BioCLIP 检索
# =========================
@torch.no_grad()
def encode_single_image(model, preprocess, image_path: Path, device: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)
    feat = model.encode_image(img_t)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)


def simplify_topk_result(
    rank: int,
    faiss_i: int,
    sim: float,
    id_map: List[str],
    id2meta: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if int(faiss_i) < 0:
        return None

    img_id = id_map[int(faiss_i)]
    meta = id2meta.get(img_id, {})

    return {
        "rank": rank,
        "id": img_id,
        "similarity": float(sim),
        "labels": meta.get("preferred_labels", [])
    }


def build_retrieval_module(
    sims: np.ndarray,
    idxs: np.ndarray,
    id_map: List[str],
    id2meta: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    topk_results: List[Dict[str, Any]] = []

    for rank, (faiss_i, sim) in enumerate(zip(idxs, sims), start=1):
        item = simplify_topk_result(rank, int(faiss_i), float(sim), id_map, id2meta)
        if item is not None:
            topk_results.append(item)

    best_idx = int(idxs[0]) if len(idxs) > 0 and int(idxs[0]) >= 0 else None
    best_sim = float(sims[0]) if len(sims) > 0 else None

    retrieval_module = {
        "enabled": True,
        "db_hit": False,
        "threshold": THRESHOLD,
        "top1_similarity": best_sim,
        "top1_id": None,
        "top1_labels": [],
        "topk": topk_results
    }

    if best_idx is None:
        return retrieval_module

    top1_id = id_map[best_idx]
    top1_meta = id2meta.get(top1_id, {})

    retrieval_module["top1_id"] = top1_id
    retrieval_module["top1_labels"] = top1_meta.get("preferred_labels", [])
    retrieval_module["db_hit"] = (best_sim is not None and best_sim >= THRESHOLD)

    return retrieval_module


def run_faiss_query(image_path: Path) -> Dict[str, Any]:
    global MODEL, PREPROCESS, INDEX, ID_MAP, ID2META, RUNTIME_DEVICE

    if MODEL is None or PREPROCESS is None or INDEX is None or ID_MAP is None or ID2META is None:
        raise RuntimeError("Service not initialized correctly")

    q_feat = encode_single_image(MODEL, PREPROCESS, image_path, device=RUNTIME_DEVICE)
    sims, idxs = INDEX.search(q_feat, TOPK)
    sims, idxs = sims[0], idxs[0]

    retrieval_module = build_retrieval_module(sims, idxs, ID_MAP, ID2META)

    top1_meta = {}
    if retrieval_module["top1_id"] is not None:
        top1_meta = ID2META.get(retrieval_module["top1_id"], {})

    return {
        "device": RUNTIME_DEVICE,
        "retrieval_module": retrieval_module,
        "top1_meta": top1_meta
    }


# =========================
# Router: sonar / biological
# =========================
def _find_sonar_index(names_map: dict) -> int:
    for idx, name in names_map.items():
        if str(name).strip().lower() == "sonar":
            return int(idx)
    return 0


def load_router_model(device: str):
    model = attempt_load(ROUTER_MODEL_PATH, device=device)  
    model.eval()  
    class_names = model.names  
    sonar_index = _find_sonar_index(class_names)  # 查找 sonar 类别的索引
    transform = classify_transforms(224)  
    return model, class_names, sonar_index, transform


@torch.no_grad()
def run_router_classification(image_path: Path) -> Dict[str, Any]:
    global ROUTER_MODEL, ROUTER_CLASS_NAMES, ROUTER_SONAR_INDEX, ROUTER_TRANSFORM, RUNTIME_DEVICE

    if ROUTER_MODEL is None or ROUTER_TRANSFORM is None:
        raise RuntimeError("Router model not initialized correctly")

    img_pil = Image.open(image_path).convert("RGB")

    # 与你旧逻辑保持一致：PIL(RGB) -> numpy(BGR) -> classify_transforms
    im = np.array(img_pil)[:, :, ::-1].copy()
    img_tensor = ROUTER_TRANSFORM(im).unsqueeze(0).to(RUNTIME_DEVICE)

    out = ROUTER_MODEL(img_tensor)
    logits = out[0] if isinstance(out, (list, tuple)) else out
    probs = torch.softmax(logits, dim=1).squeeze(0)

    top1_prob, top1_idx = torch.max(probs, dim=0)
    top1_prob = float(top1_prob.item())
    top1_idx = int(top1_idx.item())

    sonar_prob = float(probs[ROUTER_SONAR_INDEX].item())
    is_sonar = sonar_prob >= ROUTER_THRESHOLD

    top1_label_raw = ROUTER_CLASS_NAMES.get(top1_idx, str(top1_idx))
    top1_label_raw = str(top1_label_raw)

    predicted_type = "sonar" if is_sonar else "biological"
    stage = "sonar_routed" if is_sonar else "bio_routed"

    router_module = {
        "enabled": True,
        "predicted_type": predicted_type,
        "confidence": round(sonar_prob if is_sonar else (1.0 - sonar_prob), 4),
        "model_name": Path(ROUTER_MODEL_PATH).name,
        "raw_top1_label": top1_label_raw,
        "raw_top1_confidence": round(top1_prob, 4),
        "sonar_probability": round(sonar_prob, 4),
        "threshold": ROUTER_THRESHOLD
    }

    return {
        "stage": stage,
        "predicted_type": predicted_type,
        "confidence": router_module["confidence"],
        "router_module": router_module
    }


# =========================
# YOLO 分类模型预测函数
# =========================
def predict_with_yolo_classifier(model, image: Image.Image, device: str, model_name: str) -> Dict[str, Any]:
    """
    使用 YOLO 分类模型进行预测
    """
    from utils.augmentations import classify_transforms
    
    # 准备图像
    im = np.array(image)[:, :, ::-1].copy()  # RGB -> BGR
    img_tensor = classify_transforms(224)(im).unsqueeze(0).to(device)
    
    # 推理
    output = model(img_tensor)
    pred = output[0].pred[0] if hasattr(output[0], 'pred') else output[0]
    
    # 处理结果
    probs = torch.softmax(pred, dim=0)
    top1_prob, top1_idx = probs.max(0)
    
    # 获取所有类别的概率
    all_probs = probs.cpu().numpy()
    all_labels = [model.names[i] for i in range(len(all_probs))]
    
    return {
        "model_type": "yolo",
        "model_name": model_name,
        "primary_label": model.names[top1_idx.item()],
        "confidence": top1_prob.item(),
        "all_labels": all_labels,
        "all_probabilities": all_probs.tolist()
    }


# =========================
# BioCLIP2 相关函数（双编码器模型）- 修复版本
# =========================
def load_bioclip2_model(checkpoint_path: str, device: str):
    """
    加载 BioCLIP2 微调模型（双编码器架构）- 使用标准 open_clip
    """
    try:
        import open_clip
        print(f"[BioCLIP2] Loading model from {checkpoint_path} using standard open_clip")
        # 先创建模型结构，不加载预训练权重
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=False)
        # 然后手动加载checkpoint
        print(f"[BioCLIP2] Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"[BioCLIP2] Checkpoint loaded, type: {type(checkpoint)}")
        
        # 处理不同的checkpoint格式
        if 'state_dict' in checkpoint:
            print(f"[BioCLIP2] Loading from 'state_dict' key")
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            print(f"[BioCLIP2] Loading from 'model_state_dict' key")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 假设整个checkpoint就是state_dict
            print(f"[BioCLIP2] Loading checkpoint as direct state_dict")
            model.load_state_dict(checkpoint)
        
        model = model.to(device).eval()
        print(f"[BioCLIP2] Model loaded successfully")
        
        # 返回 model, preprocess, tokenize 函数
        return model, preprocess, open_clip.tokenize
        
    except Exception as e:
        print(f"\n[BioCLIP2] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

def extract_terms_from_shard(shard_path: str, max_terms: int = 100) -> List[str]:
    """
    从 webdataset shard 中提取生物术语
    """
    print(f"\n[DEBUG] extract_terms_from_shard received: '{shard_path}'")
    
    # 先检查文件是否存在
    import os
    if not os.path.exists(shard_path):
        print(f"\n[BioCLIP2] Shard file not found: {shard_path}")
        # 返回默认术语，避免WebDataset的内部fallback
        return get_default_terms()
    
    try:
        import webdataset as wds
    except ImportError:
        print("[BioCLIP2] webdataset not installed, using default terms")
        return get_default_terms()
    
    terms = set()
    try:
        # 关键修改：使用列表形式明确指定单个文件
        # 这可以避免 WebDataset 自动添加 .tar 后缀的问题
        print(f"[BioCLIP2] Extracting terms from {shard_path} using list format")
        
        dataset = wds.WebDataset([shard_path])  # 用列表形式
        
        # 遍历数据集
        for sample in dataset:
            # 查找文本文件
            for key in ['sci.txt', 'com.txt', 'taxon.txt']:
                if key in sample:
                    try:
                        term = sample[key].decode('utf-8', errors='ignore').strip()
                        if term and len(term) < 100:  # 过滤掉过长的术语
                            terms.add(term)
                    except:
                        pass
            if len(terms) > max_terms:
                break
                
        print(f"[BioCLIP2] Extracted {len(terms)} terms")
        
        if terms:
            return list(terms)[:max_terms]
        else:
            print("[BioCLIP2] No terms extracted, using default terms")
            return get_default_terms()
            
    except Exception as e:
        print(f"[BioCLIP2] Error extracting terms: {e}")
        import traceback
        traceback.print_exc()
        return get_default_terms()

def get_default_terms() -> List[str]:
    """返回默认的生物术语列表"""
    return [
        'coral', 'fish', 'marine', 'invertebrate', 
        'Acanthastrea', 'Acropora', 'Alveopora', 'Amphiprion',
        'Canthigaster', 'Carcharhinus', 'Catalaphyllia', 'Chaetodon',
        'Chelmon', 'Chromis', 'Cleidopus', 'Colpophyllia',
        'Coradion', 'Cyphastrea', 'Dendrogyra', 'Discosoma',
        'Euphyllia', 'Forcipiger', 'Heliofungia', 'Heniochus',
        'Hydnophora', 'Leptoseris', 'Meandrina', 'Millepora',
        'Monocentris', 'Montastraea', 'Montipora', 'Myripristis',
        'Naso', 'Palythoa', 'Paraluteres', 'Platax',
        'Pocillopora', 'Porites', 'Pseudanthias', 'Rhodactis',
        'Ricordea', 'Seriatopora', 'Siganus', 'Toxotes'
    ]

def load_bioclip2_text_features(model, terms: List[str], device: str):
    """
    预计算文本特征 - 使用标准 open_clip
    """
    import open_clip
    
    text_tokens = open_clip.tokenize(terms).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features, terms


@torch.no_grad()
def predict_with_bioclip2(
    image: Image.Image, 
    model, 
    preprocess, 
    text_features: torch.Tensor, 
    terms: List[str],
    device: str,
    topk: int = 5
) -> Dict[str, Any]:
    """
    使用 BioCLIP2 微调模型进行预测（图像-文本匹配）
    """
    # 预处理图像
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # 编码图像
    img_features = model.encode_image(img_tensor)
    img_features /= img_features.norm(dim=-1, keepdim=True)
    
    # 计算相似度
    similarity = (img_features @ text_features.T).squeeze(0)
    scores, indices = similarity.topk(min(topk, len(terms)))
    
    # 整理结果
    matches = []
    for i in range(len(indices)):
        idx = indices[i].item()
        matches.append({
            "term": terms[idx],
            "similarity": scores[i].item()
        })
    
    # 判断是鱼还是珊瑚（根据匹配的术语）
    top_term = matches[0]["term"].lower() if matches else ""
    fish_keywords = ['fish', 'pisces', 'ichthys', 'shark', 'ray', 'eel', 'grouper', 'snapper', 
                     'amphiprion', 'canthigaster', 'carcharhinus', 'chaetodon', 'chelmon', 
                     'chromis', 'cleidopus', 'coradion', 'forcipiger', 'heniochus', 'monocentris',
                     'myripristis', 'naso', 'paraluteres', 'platax', 'pseudanthias', 'siganus', 'toxotes']
    coral_keywords = ['coral', 'anthozoa', 'acropora', 'alveopora', 'catalaphyllia', 'colpophyllia',
                      'cyphastrea', 'dendrogyra', 'discosoma', 'euphyllia', 'heliofungia', 'hydnophora',
                      'leptoseris', 'meandrina', 'millepora', 'montastraea', 'montipora', 'palythoa',
                      'pocillopora', 'porites', 'rhodactis', 'ricordea', 'seriatopora']
    
    is_fish = any(keyword in top_term for keyword in fish_keywords)
    is_coral = any(keyword in top_term for keyword in coral_keywords)
    
    # 如果无法从关键词判断，看匹配的术语列表
    if not is_fish and not is_coral:
        # 检查前3个匹配中是否有明显的鱼或珊瑚关键词
        for match in matches[:3]:
            term_lower = match["term"].lower()
            if any(keyword in term_lower for keyword in fish_keywords):
                is_fish = True
                break
            if any(keyword in term_lower for keyword in coral_keywords):
                is_coral = True
                break
    
    return {
        "model_type": "bioclip2",
        "matches": matches,
        "primary_match": matches[0] if matches else None,
        "is_fish": is_fish,
        "is_coral": is_coral,
        "confidence": matches[0]["similarity"] if matches else 0.0
    }

# =========================
# 主流程
# =========================
def run_full_pipeline(image_path: Path) -> Dict[str, Any]:
    global FISH_MODEL, CORAL_MODEL, SONAR_MODEL, ROUTER_MODEL, FISH_CORAL_MODEL
    global BIOCLIP2_MODEL, BIOCLIP2_PREPROCESS, BIOCLIP2_TOKENIZER
    global BIOCLIP2_TEXT_FEATURES, BIOCLIP2_TERMS, USE_BIOCLIP2
    global RUNTIME_DEVICE
    
    # 先加载图像（只加载一次）
    image = Image.open(image_path).convert("RGB")
    
    faiss_result = run_faiss_query(image_path)
    retrieval_module = faiss_result["retrieval_module"]
    top1_meta = faiss_result["top1_meta"]

    modules = build_default_modules()
    modules["retrieval"] = retrieval_module

    # 1. 数据库命中，直接返回
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
            note="Matched from retrieval database."
        )

        return {
            "success": True,
            "stage": "db_hit",
            "final_result": final_result,
            "modules": modules
        }

    # 2. 数据库未命中，进入 router
    router_result = run_router_classification(image_path)
    modules["router"] = router_result["router_module"]

    image_type = router_result["predicted_type"]
    stage = router_result["stage"]
    confidence = router_result["confidence"]

    # 3. 根据路由结果分流
    if image_type == "sonar":
        # 声纳图 - 使用声纳分类模型
        result = predict_with_yolo_classifier(SONAR_MODEL, image, RUNTIME_DEVICE, "MergedData_7")
        modules["sonar"] = result
        
        final_result = build_final_result(
            status="success",
            source="sonar_model",
            image_type="sonar",
            primary_label=result["primary_label"],
            all_labels=result["all_labels"],
            confidence=result["confidence"],
            display_text=f"声纳图像识别：{result['primary_label']}",
            note="Sonar classification completed."
        )
        
    else:  # biological
        # 先使用 fish/coral 二分类模型
        fish_coral_result = predict_with_yolo_classifier(
            FISH_CORAL_MODEL, image, RUNTIME_DEVICE, "fish_coral_cls"
        )
        modules["fish_coral_classifier"] = fish_coral_result
        
        # 根据配置选择使用 BioCLIP2 还是 YOLO
        if USE_BIOCLIP2 and BIOCLIP2_MODEL is not None and BIOCLIP2_TEXT_FEATURES is not None:
            # 用 BioCLIP2 进行图像-文本匹配识别
            result = predict_with_bioclip2(
                image=image,
                model=BIOCLIP2_MODEL,
                preprocess=BIOCLIP2_PREPROCESS,
                text_features=BIOCLIP2_TEXT_FEATURES,
                terms=BIOCLIP2_TERMS,
                device=RUNTIME_DEVICE,
                topk=5
            )
            
            modules["bioclip2"] = result
            
            # 根据匹配结果判断是鱼还是珊瑚
            if result["is_fish"]:
                source = "bioclip2_fish"
                img_type = "fish"
                display_prefix = "鱼类"
            elif result["is_coral"]:
                source = "bioclip2_coral"
                img_type = "coral"
                display_prefix = "珊瑚"
            else:
                source = "bioclip2_unknown"
                img_type = "biological_unknown"
                display_prefix = "生物"
            
            # 构建显示文本
            if result["primary_match"]:
                display_text = f"{display_prefix}识别(BioCLIP2)：{result['primary_match']['term']} (相似度: {result['primary_match']['similarity']:.3f})"
                primary_label = result['primary_match']['term']
                all_labels = [m['term'] for m in result['matches']]
                conf = result['confidence']
            else:
                display_text = f"{display_prefix}识别(BioCLIP2)：无匹配结果"
                primary_label = None
                all_labels = []
                conf = None
            
            final_result = build_final_result(
                status="success",
                source=source,
                image_type=img_type,
                primary_label=primary_label,
                all_labels=all_labels,
                confidence=conf,
                display_text=display_text,
                note=f"{img_type} classification with BioCLIP2 image-text matching."
            )
            
        else:
            # 用 YOLO 模型，根据二分类结果选择
            if fish_coral_result["primary_label"].lower() == "fish":
                # 鱼类识别
                result = predict_with_yolo_classifier(
                    FISH_MODEL, image, RUNTIME_DEVICE, "merge_fish_small"
                )
                modules["fish"] = result
                
                final_result = build_final_result(
                    status="success",
                    source="fish_model",
                    image_type="fish",
                    primary_label=result["primary_label"],
                    all_labels=result["all_labels"],
                    confidence=result["confidence"],
                    display_text=f"鱼类识别：{result['primary_label']}",
                    note="Fish classification with YOLO."
                )
                
            else:  # coral
                # 珊瑚识别
                result = predict_with_yolo_classifier(
                    CORAL_MODEL, image, RUNTIME_DEVICE, "Coral_one2"
                )
                modules["coral"] = result
                
                final_result = build_final_result(
                    status="success",
                    source="coral_model",
                    image_type="coral",
                    primary_label=result["primary_label"],
                    all_labels=result["all_labels"],
                    confidence=result["confidence"],
                    display_text=f"珊瑚识别：{result['primary_label']}",
                    note="Coral classification with YOLO."
                )

    return {
        "success": True,
        "stage": stage,
        "final_result": final_result,
        "modules": modules
    }


# =========================
# FastAPI 生命周期
# =========================
@app.on_event("startup")
def startup_event():
    global MODEL, PREPROCESS, INDEX, ID_MAP, ID2META, RUNTIME_DEVICE
    global ROUTER_MODEL, ROUTER_CLASS_NAMES, ROUTER_SONAR_INDEX, ROUTER_TRANSFORM
    global SONAR_MODEL, FISH_MODEL, CORAL_MODEL, FISH_CORAL_MODEL
    global BIOCLIP2_MODEL, BIOCLIP2_PREPROCESS, BIOCLIP2_TOKENIZER
    global BIOCLIP2_TEXT_FEATURES, BIOCLIP2_TERMS

    try:
        RUNTIME_DEVICE = resolve_device(DEVICE)
        print(f"[Startup] Using device: {RUNTIME_DEVICE}")

        # 1. 加载 BioCLIP 检索模型
        MODEL, PREPROCESS = load_bioclip(RUNTIME_DEVICE)
        print("[Startup] BioCLIP retrieval model loaded.")

        # 2. 加载 FAISS 索引
        INDEX, ID_MAP = load_index()
        print("[Startup] FAISS index loaded.")

        # 3. 加载元数据
        ID2META = load_metadata()
        print("[Startup] Metadata loaded.")

        # 4. 加载 Router 模型 (sonar/生物 二分类)
        print("[Startup] Loading router model...")
        ROUTER_MODEL, ROUTER_CLASS_NAMES, ROUTER_SONAR_INDEX, ROUTER_TRANSFORM = load_router_model(RUNTIME_DEVICE)
        print(f"[Startup] Router model loaded. Classes: {ROUTER_CLASS_NAMES}")

        # 5. 加载声纳图分类模型
        print("[Startup] Loading sonar classification model...")
        SONAR_MODEL = attempt_load(SONAR_CLS_PATH, device=RUNTIME_DEVICE)
        SONAR_MODEL.eval()
        print(f"[Startup] Sonar model loaded. Classes: {SONAR_MODEL.names}")

        # 6. 加载 fish/coral 二分类模型
        print("[Startup] Loading fish/coral classifier...")
        FISH_CORAL_MODEL = attempt_load(FISH_CORAL_CLS_PATH, device=RUNTIME_DEVICE)
        FISH_CORAL_MODEL.eval()
        print(f"[Startup] Fish/Coral model loaded. Classes: {FISH_CORAL_MODEL.names}")

        # 7. 加载鱼类分类模型
        print("[Startup] Loading fish classification model...")
        FISH_MODEL = attempt_load(FISH_MODEL_PATH, device=RUNTIME_DEVICE)
        FISH_MODEL.eval()
        print(f"[Startup] Fish model loaded. Classes: {FISH_MODEL.names}")

        # 8. 加载珊瑚分类模型
        print("[Startup] Loading coral classification model...")
        CORAL_MODEL = attempt_load(CORAL_MODEL_PATH, device=RUNTIME_DEVICE)
        CORAL_MODEL.eval()
        print(f"[Startup] Coral model loaded. Classes: {CORAL_MODEL.names}")

        # 9. 加载 BioCLIP2 微调模型（双编码器）
        print("[Startup] Loading BioCLIP2 finetuned model...")
        try:
            BIOCLIP2_MODEL, BIOCLIP2_PREPROCESS, BIOCLIP2_TOKENIZER = load_bioclip2_model(
                BIOCLIP2_CHECKPOINT, RUNTIME_DEVICE
            )
            
            # 10. 从 shard 提取术语并预计算文本特征
            if BIOCLIP2_MODEL is not None:
                print(f"\n[DEBUG] SHARD_PATH = '{SHARD_PATH}'")
                print(f"\n[DEBUG] SHARD_PATH type = {type(SHARD_PATH)}")
                print(f"\n[DEBUG] File exists: {os.path.exists(SHARD_PATH)}")
                BIOCLIP2_TERMS = extract_terms_from_shard(SHARD_PATH, max_terms=100)
                if BIOCLIP2_TERMS:
                    BIOCLIP2_TEXT_FEATURES, BIOCLIP2_TERMS = load_bioclip2_text_features(
                        BIOCLIP2_MODEL, BIOCLIP2_TERMS, RUNTIME_DEVICE
                    )
                    print(f"[Startup] BioCLIP2 text features computed for {len(BIOCLIP2_TERMS)} terms")
                else:
                    print("[Startup] Warning: No terms extracted for BioCLIP2")
            else:
                print("\nBIOCLIP2_MODEL is None ")
        except Exception as e:
            print(f"[Startup] Warning: Failed to load BioCLIP2: {e}")
            BIOCLIP2_MODEL = None
            BIOCLIP2_PREPROCESS = None
            BIOCLIP2_TOKENIZER = None
            BIOCLIP2_TEXT_FEATURES = None
            BIOCLIP2_TERMS = []
        
    except Exception as e:
        print(f"[Startup][ERROR] {e}")
        raise


# =========================
# 路由
# =========================
@app.get("/health")
def health():
    ready = all([
        MODEL is not None,
        PREPROCESS is not None,
        INDEX is not None,
        ID_MAP is not None,
        ID2META is not None,
        RUNTIME_DEVICE is not None,
        ROUTER_MODEL is not None,
        ROUTER_TRANSFORM is not None,
        SONAR_MODEL is not None,
        FISH_CORAL_MODEL is not None,
        FISH_MODEL is not None,
        CORAL_MODEL is not None
    ])

    health_info = {
        "success": ready,
        "service": APP_TITLE,
        "version": APP_VERSION,
        "device": RUNTIME_DEVICE,
        "ready": ready,
        "router_model_loaded": ROUTER_MODEL is not None,
        "router_classes": ROUTER_CLASS_NAMES,
        "router_sonar_index": ROUTER_SONAR_INDEX,
        "sonar_model_loaded": SONAR_MODEL is not None,
        "fish_coral_model_loaded": FISH_CORAL_MODEL is not None,
        "fish_model_loaded": FISH_MODEL is not None,
        "coral_model_loaded": CORAL_MODEL is not None,
        "bioclip2_loaded": BIOCLIP2_MODEL is not None,
        "bioclip2_terms_count": len(BIOCLIP2_TERMS) if BIOCLIP2_TERMS else 0,
        "use_bioclip2": USE_BIOCLIP2
    }
    
    return health_info


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    request_id = generate_request_id()
    suffix = Path(file.filename).suffix if file.filename else ".jpg"

    tmp_dir = tempfile.mkdtemp(prefix="predict_")
    tmp_path = Path(tmp_dir) / f"input{suffix}"

    try:
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        result = run_full_pipeline(tmp_path)

        return JSONResponse(content={
            "success": True,
            "request_id": request_id,
            "message": "Prediction completed.",
            "filename": file.filename,
            "content_type": file.content_type,
            "result": result
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "request_id": request_id,
                "message": f"Inference error: {str(e)}"
            }
        )
    finally:
        try:
            file.file.close()
        except Exception:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run(
        "app_success:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )