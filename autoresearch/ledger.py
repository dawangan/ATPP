"""Append-only, hash-chained ledger of everything the harness does.

Why: the dominant failure mode of automated research is the garden of forking paths: many
specifications tried, the best one reported. The ledger records EVERY run (including ones that
fail or are rejected) so the report can state how many trials produced a headline number, and
so multiplicity can be priced in. Each entry commits to the previous one (sha256 chain), so
silent edits or deletions are detectable with `python -m autoresearch ledger-verify`.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from .core import STATE, canon

try:                                    # POSIX cross-process lock; absent on Windows
    import fcntl
except ImportError:                     # pragma: no cover
    fcntl = None

GENESIS = "0" * 64


class Ledger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("AUTORESEARCH_LEDGER") or STATE / "ledger.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ write
    def append(self, kind: str, **fields) -> dict:
        with self._lock, open(self.path, "r+") as f:
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                last = None
                for line in f:
                    if line.strip():
                        last = line
                prev = json.loads(last)["hash"] if last else GENESIS
                seq = (json.loads(last)["seq"] + 1) if last else 0
                body = {"seq": seq, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "kind": kind, "prev": prev, **fields}
                body["hash"] = hashlib.sha256((prev + canon(body)).encode()).hexdigest()
                f.seek(0, 2)
                f.write(json.dumps(body, default=str) + "\n")
                f.flush()
                return body
            finally:
                if fcntl:
                    fcntl.flock(f, fcntl.LOCK_UN)

    # ------------------------------------------------------------------- read
    def entries(self, kind: str | None = None) -> list[dict]:
        out = []
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    if kind is None or e["kind"] == kind:
                        out.append(e)
        return out

    def verify(self) -> tuple[bool, int | None]:
        """(ok, first_bad_seq). Recomputes the whole chain."""
        prev = GENESIS
        for e in self.entries():
            claimed = e["hash"]
            body = {k: v for k, v in e.items() if k != "hash"}
            if e["prev"] != prev or hashlib.sha256((prev + canon(body)).encode()).hexdigest() != claimed:
                return False, e["seq"]
            prev = claimed
        return True, None

    # -------------------------------------------------------------- multiplicity
    def spec_trials(self, family: str) -> list[str]:
        """Distinct spec hashes ever run for a hypothesis family (seed excluded: a re-run of the
        same spec is a replication, a different spec is another look at the data)."""
        seen: list[str] = []
        for e in self.entries("experiment"):
            if e.get("family") == family and e.get("data_kind") != "synthetic" and e["spec_hash"] not in seen:
                seen.append(e["spec_hash"])
        return seen
