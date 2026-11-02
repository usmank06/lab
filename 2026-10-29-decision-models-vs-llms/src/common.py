"""Shared types and helpers: config loading, question schema, paths, hashing."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
CONFIG_PATH = ROOT / "config.yaml"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"
DB_PATH = DATA_DIR / "runs.sqlite"

FORMATS = ("yesno", "choice", "score")
TIERS = (1, 2, 3)
SCORE_LEVELS = 5
LETTERS = "ABCDEFGH"


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stable_hash(*parts: Any, bits: int = 64) -> int:
    """Deterministic integer hash of the given parts (for per-question seeds)."""
    h = hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h, 16) % (2**bits)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Question:
    id: str
    tier: int
    format: str                       # yesno | choice | score
    state: Any                        # str, or JSON-serialisable object
    question: str
    options: list[str] | None         # choice: 4-8 option texts; score: 5 ordered level labels (low->high); yesno: None
    ground_truth: Any                 # yesno: "yes"/"no"; choice: option index (int); score: level index 0..4 (int)
    domain: str
    generator: str
    generator_seed: int
    ground_truth_text: str = ""
    difficulty: int | None = None     # tier 2: number of constraints / entities
    in_labelswap: bool = False
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Question":
        return Question(**d)

    def state_text(self) -> str:
        """The state rendered as text (JSON states are pretty-printed)."""
        if isinstance(self.state, str):
            return self.state
        return json.dumps(self.state, indent=2, ensure_ascii=False)

    def n_options(self) -> int:
        if self.format == "yesno":
            return 2
        return len(self.options or [])


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions(path: Path | None = None) -> list[Question]:
    return [Question.from_dict(d) for d in read_jsonl(path or QUESTIONS_PATH)]


ENV_FILE = ROOT / ".env"


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """KEY=VALUE lines from a local .env (never published; see publish.ps1 and .gitignore)."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def get_api_key(cfg: dict) -> str | None:
    env = cfg["openrouter"].get("api_key_env", "OPENROUTER_API_KEY")
    key = os.environ.get(env) or read_env_file().get(env)
    return key or None


def scrub_secret(obj: Any, secret: str | None) -> Any:
    """Recursively replace any occurrence of the secret in strings (defensive; raw files must never contain it)."""
    if not secret:
        return obj
    if isinstance(obj, str):
        return obj.replace(secret, "[REDACTED]")
    if isinstance(obj, list):
        return [scrub_secret(x, secret) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_secret(v, secret) for k, v in obj.items()}
    return obj
