from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from PIL import Image


def load_bioclip2_model(checkpoint_path: str, device: str):
    try:
        import open_clip

        print(f"[BioCLIP2] Loading model from {checkpoint_path} using standard open_clip")
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained=False)

        print("[BioCLIP2] Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"[BioCLIP2] Checkpoint loaded, type: {type(checkpoint)}")

        if "state_dict" in checkpoint:
            print("[BioCLIP2] Loading from 'state_dict' key")
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            print("[BioCLIP2] Loading from 'model_state_dict' key")
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("[BioCLIP2] Loading checkpoint as direct state_dict")
            model.load_state_dict(checkpoint)

        model = model.to(device).eval()
        print("[BioCLIP2] Model loaded successfully")

        return model, preprocess, open_clip.tokenize

    except Exception as e:
        print(f"\n[BioCLIP2] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_default_terms() -> List[str]:
    return [
        "coral", "fish", "marine", "invertebrate",
        "Acanthastrea", "Acropora", "Alveopora", "Amphiprion",
        "Canthigaster", "Carcharhinus", "Catalaphyllia", "Chaetodon",
        "Chelmon", "Chromis", "Cleidopus", "Colpophyllia",
        "Coradion", "Cyphastrea", "Dendrogyra", "Discosoma",
        "Euphyllia", "Forcipiger", "Heliofungia", "Heniochus",
        "Hydnophora", "Leptoseris", "Meandrina", "Millepora",
        "Monocentris", "Montastraea", "Montipora", "Myripristis",
        "Naso", "Palythoa", "Paraluteres", "Platax",
        "Pocillopora", "Porites", "Pseudanthias", "Rhodactis",
        "Ricordea", "Seriatopora", "Siganus", "Toxotes",
    ]


def extract_terms_from_shard(shard_path: str, max_terms: int = 100) -> List[str]:
    print(f"\n[DEBUG] extract_terms_from_shard received: '{shard_path}'")

    if not os.path.exists(shard_path):
        print(f"\n[BioCLIP2] Shard file not found: {shard_path}")
        return get_default_terms()

    try:
        import webdataset as wds
    except ImportError:
        print("[BioCLIP2] webdataset not installed, using default terms")
        return get_default_terms()

    terms = set()
    try:
        print(f"[BioCLIP2] Extracting terms from {shard_path} using list format")
        dataset = wds.WebDataset([shard_path])

        for sample in dataset:
            for key in ["sci.txt", "com.txt", "taxon.txt"]:
                if key in sample:
                    try:
                        term = sample[key].decode("utf-8", errors="ignore").strip()
                        if term and len(term) < 100:
                            terms.add(term)
                    except Exception:
                        pass

            if len(terms) > max_terms:
                break

        print(f"[BioCLIP2] Extracted {len(terms)} terms")

        if terms:
            return list(terms)[:max_terms]

        print("[BioCLIP2] No terms extracted, using default terms")
        return get_default_terms()

    except Exception as e:
        print(f"[BioCLIP2] Error extracting terms: {e}")
        import traceback
        traceback.print_exc()
        return get_default_terms()


def load_bioclip2_text_features(model, terms: List[str], device: str):
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
    topk: int = 5,
) -> Dict[str, Any]:
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    img_features = model.encode_image(img_tensor)
    img_features /= img_features.norm(dim=-1, keepdim=True)

    similarity = (img_features @ text_features.T).squeeze(0)
    scores, indices = similarity.topk(min(topk, len(terms)))

    matches = []
    for i in range(len(indices)):
        idx = indices[i].item()
        matches.append({
            "term": terms[idx],
            "similarity": scores[i].item(),
        })

    top_term = matches[0]["term"].lower() if matches else ""

    fish_keywords = [
        "fish", "pisces", "ichthys", "shark", "ray", "eel", "grouper", "snapper",
        "amphiprion", "canthigaster", "carcharhinus", "chaetodon", "chelmon",
        "chromis", "cleidopus", "coradion", "forcipiger", "heniochus", "monocentris",
        "myripristis", "naso", "paraluteres", "platax", "pseudanthias", "siganus", "toxotes",
    ]
    coral_keywords = [
        "coral", "anthozoa", "acropora", "alveopora", "catalaphyllia", "colpophyllia",
        "cyphastrea", "dendrogyra", "discosoma", "euphyllia", "heliofungia", "hydnophora",
        "leptoseris", "meandrina", "millepora", "montastraea", "montipora", "palythoa",
        "pocillopora", "porites", "rhodactis", "ricordea", "seriatopora",
    ]

    is_fish = any(keyword in top_term for keyword in fish_keywords)
    is_coral = any(keyword in top_term for keyword in coral_keywords)

    if not is_fish and not is_coral:
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
        "confidence": matches[0]["similarity"] if matches else 0.0,
    }
