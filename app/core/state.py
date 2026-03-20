from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AppState:
    model: Any = None
    preprocess: Any = None
    index: Any = None
    id_map: Optional[List[str]] = None
    id2meta: Optional[Dict[str, Dict[str, Any]]] = None
    runtime_device: Optional[str] = None

    router_model: Any = None
    router_class_names: Dict[int, str] = field(default_factory=dict)
    router_sonar_index: int = 0
    router_transform: Any = None

    sonar_model: Any = None
    fish_coral_model: Any = None
    fish_model: Any = None
    coral_model: Any = None

    bioclip2_model: Any = None
    bioclip2_preprocess: Any = None
    bioclip2_tokenizer: Any = None
    bioclip2_text_features: Any = None
    bioclip2_terms: List[str] = field(default_factory=list)


state = AppState()
