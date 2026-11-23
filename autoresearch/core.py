"""Paths, hashing and small shared helpers."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # repo root (contains l1/, scripts/)
PKG = Path(__file__).resolve().parent
STATE = PKG / "state"
PROMPTS = PKG / "prompts"


def canon(obj) -> str:
    """Canonical JSON: sorted keys, no whitespace, numpy-safe via default=str."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(obj) -> str:
    return hashlib.sha256(canon(obj).encode()).hexdigest()


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def code_hash(subdirs=("l1", "scripts")) -> str:
    """Hash of the analysis code a result depends on (so a result is tied to the code that made it)."""
    h = hashlib.sha256()
    for d in subdirs:
        for p in sorted((ROOT / d).rglob("*.py")):
            h.update(str(p.relative_to(ROOT)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"sha": run("rev-parse", "HEAD") or "unknown", "dirty": bool(run("status", "--porcelain"))}


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def dump_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str))
