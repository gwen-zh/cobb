from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
ASSAYS = json.loads((ROOT / "assays.json").read_text())
BY_ID = {assay["id"]: assay for assay in ASSAYS}
AA = "ACDEFGHIKLMNPQRSTVWY"
MODEL_SPECS = {
    "E": ("biohub/esm2_t12_35M_UR50D", "6fbf070e65b0b7291e7bbcd451118c216cff79d8", "model.safetensors"),
    "P": ("westlake-repl/SaProt_35M_AF2", "316cd4017d29f4657b959365f24b57f1ee278912", "pytorch_model.bin"),
    "Q": ("westlake-repl/SaProt_35M_AF2_seqOnly", "6628b8a5288695045fdb4575a361e84fdfefe5c8", "pytorch_model.bin"),
}
FEATURE_SETS = ("D", "D+S", "D+E", "D+P", "D+E+S", "D+E+P", "D+E+P+S")
MODELS = ("Ridge", "GBT", "MLP")
SPLITS = ("random", "position")
BUDGETS = (50, 100, 250, 500)
SEEDS = (17, 29, 43, 71, 97)
RSA_CUTOFF = 0.20


def selected_assays(names: list[str] | None) -> list[dict]:
    if not names:
        return ASSAYS
    aliases = {item["label"].lower(): item for item in ASSAYS}
    aliases.update({item["id"].lower(): item for item in ASSAYS})
    return [aliases[name.lower()] for name in names]


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(obj, indent=2) + "\n")
    staging.replace(path)
