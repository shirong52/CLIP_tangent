import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


class ControlledPairsDataset(Dataset):
    """Dataset for controlled paired records with original and perturbed examples."""

    def __init__(
        self,
        jsonl_path: str,
        dataset_root: Optional[str] = None,
        factors: Optional[List[str]] = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"jsonl file not found: {self.jsonl_path}")

        self.dataset_root = Path(dataset_root) if dataset_root else self.jsonl_path.parent
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"dataset root not found: {self.dataset_root}")

        self.records: List[Dict] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if factors and item.get("factor") not in factors:
                    continue
                self.records.append(item)

        if not self.records:
            raise ValueError("No valid records loaded from dataset.")

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rel_path: str) -> Image.Image:
        full_path = self.dataset_root / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
        return Image.open(full_path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        image = self._load_image(record["image"])
        image_plus = self._load_image(record["image_plus"])

        return {
            "pair_id": record.get("pair_id", str(idx)),
            "factor": record.get("factor", "unknown"),
            "image": image,
            "image_plus": image_plus,
            "caption": record["caption"],
            "caption_plus": record["caption_plus"],
        }
