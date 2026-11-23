"""JSON schemas the LLM roles must satisfy (enforced by `claude --json-schema`, re-validated here)."""
from __future__ import annotations

LANES = ["experiment", "literature", "code_audit"]

_FOLLOWUP = {"type": "object", "required": ["question", "why", "lane"],
             "properties": {"question": {"type": "string"}, "why": {"type": "string"},
                            "lane": {"type": "string", "enum": LANES},
                            "experiment": {"type": ["string", "null"]}}}

FINDING = {
    "type": "object",
    "required": ["claim", "kind", "claims_effect", "confidence", "evidence", "caveats", "followups"],
    "properties": {
        "claim": {"type": "string"},
        "kind": {"type": "string", "enum": ["effect", "null", "method", "novelty", "defect", "inconclusive"]},
        "claims_effect": {"type": "boolean",
                          "description": "true iff the claim asserts a real effect / that a gate passes"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "evidence": {"type": "array", "items": {"type": "object", "required": ["ref", "note"],
                     "properties": {"ref": {"type": "string"}, "note": {"type": "string"}}}},
        "caveats": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {"type": "object", "required": ["claim", "url", "quote"],
                      "properties": {"claim": {"type": "string"}, "url": {"type": "string"}, "quote": {"type": "string"}}}},
        "searches": {"type": "array", "items": {"type": "string"}},
        "defects": {"type": "array", "items": {"type": "object", "required": ["file", "line", "severity", "why", "repro_code", "repro_kind"],
                    "properties": {"file": {"type": "string"}, "line": {"type": "integer"},
                                   "severity": {"type": "string", "enum": ["minor", "major", "critical"]},
                                   "why": {"type": "string"},
                                   "repro_kind": {"type": "string", "enum": ["behavioral", "structural"],
                                                  "description": "behavioral = shows a wrong output; structural = shows a code fact (e.g. a call is absent)"},
                                   "repro_code": {"type": "string",
                                                  "description": "standalone python; must raise AssertionError iff the bug is present"}}}},
        "followups": {"type": "array", "items": _FOLLOWUP},
    },
}

REVIEW = {
    "type": "object",
    "required": ["lens", "verdict", "issues", "what_would_change_my_mind"],
    "properties": {
        "lens": {"type": "string"},
        "verdict": {"type": "string", "enum": ["pass", "concern", "block"]},
        "issues": {"type": "array", "items": {"type": "object", "required": ["severity", "claim", "evidence", "suggested_fix"],
                   "properties": {"severity": {"type": "string", "enum": ["minor", "major", "blocker"]},
                                  "claim": {"type": "string"}, "evidence": {"type": "string"},
                                  "suggested_fix": {"type": "string"}}}},
        "what_would_change_my_mind": {"type": "string"},
    },
}

FINAL = {
    "type": "object",
    "required": ["summary", "ranked_next_steps"],
    "properties": {
        "summary": {"type": "string"},
        "ranked_next_steps": {"type": "array", "items": {"type": "object", "required": ["title", "why", "priority"],
                              "properties": {"title": {"type": "string"}, "why": {"type": "string"},
                                             "priority": {"type": "integer"}}}},
    },
}

LENSES = {"experiment": ["statistician", "timing", "skeptic"], "literature": ["citation", "skeptic"],
          "code_audit": ["skeptic"]}
