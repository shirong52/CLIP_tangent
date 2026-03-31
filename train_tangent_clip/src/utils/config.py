import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrainConfig:
    raw: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def load_config(path: str) -> TrainConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return TrainConfig(raw=data)
