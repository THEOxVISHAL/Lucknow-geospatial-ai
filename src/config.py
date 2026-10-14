"""
Lucknow Land Cover Classification - Core Configuration Loader
"""
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import torch


@dataclass
class Config:
    project: Dict[str, Any]
    data: Dict[str, Any]
    classes: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]
    inference: Dict[str, Any]
    label_generation: Dict[str, Any]

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def n_classes(self) -> int:
        return self.classes["n_classes"]

    @property
    def class_names(self) -> List[str]:
        return self.classes["names"]

    @property
    def class_colors(self) -> List[List[int]]:
        return self.classes["colors"]


def load_config(config_path: str = "configs/config.yaml") -> Config:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg_dict = yaml.safe_load(f)
    return Config(**cfg_dict)


CONFIG = load_config()