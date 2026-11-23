"""Fast, offline tests of the harness's deterministic machinery (no simulation, no LLM).

    python -m pytest autoresearch/tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autoresearch import checks, prereg as P, verdict
from autoresearch.ledger import Ledger
from autoresearch.orchestrator import run_repro, vet
from autoresearch.experiments import normalize_layer1

PR = P.load()


# ------------------------------------------------------------------ ledger
def test_ledger_chain_detects_tampering(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    for i in range(4):
        led.append("experiment", task_id=f"t{i}", family="H1_stage1", data_kind="real", spec_hash=f"h{i % 2}")
    assert led.verify() == (True, None)
    lines = led.path.read_text().splitlines()
    e = json.loads(lines[1])
    e["task_id"] = "forged"
    lines[1] = json.dumps(e)
    led.path.write_text("\n".join(lines) + "\n")
    ok, bad = led.verify()
    assert not ok and bad == 1


def test_ledger_deletion_detected(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    for i in range(3):
        led.append("experiment", task_id=str(i), family="H1_stage1", data_kind="real", spec_hash=str(i))
    lines = led.path.read_text().splitlines()
    led.path.write_text("\n".join(lines[:1] + lines[2:]) + "\n")      # drop the middle entry
    assert not led.verify()[0]


def test_spec_trials_counts_distinct_real_specs_only(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    led.append("experiment", family="H1_stage1", data_kind="real", spec_hash="a")
    led.append("experiment", family="H1_stage1", data_kind="real", spec_hash="a")        # replication, not a new look
    led.append("experiment", family="H1_stage1", data_kind="real", spec_hash="b")
    led.append("experiment", family="H1_stage1", data_kind="synthetic", spec_hash="c")   # synthetic never counts
    assert led.spec_trials("H1_stage1") == ["a", "b"]


# ------------------------------------------------------------------ prereg guards
def test_holdout_guard():
    P.assert_explore("2025-09-18", PR)
    with pytest.raises(P.HoldoutViolation):
        P.assert_explore("2025-09-19", PR)


def test_drift_detected_only_on_frozen_fields():
    assert P.drift({"half_life": 1800.0, "placebos": 5}, PR) == []
    assert P.drift({"half_life": 900.0}, PR) != []


# ------------------------------------------------------------------ verdict
def _res(**kw):
    base = {"estimates": {"scheduled": {"gamma": -0.03}, "surprise": {"gamma": -0.02}},
            "wald_robust": {"p": 1e-5}, "n_placebos": 200,
            "placebo": {"scheduled": {"p": 0.004, "mean": 0.001, "sd": 0.01}, "surprise": {"p": 0.004, "mean": 0.0, "sd": 0.01}},
            "offset_scan": [{"delta_s": -600, "loglik": -5.0}, {"delta_s": 0, "loglik": -1.0}, {"delta_s": 600, "loglik": -4.0}]}
    base.update(kw)
    return base


def test_verdict_all_pass():
    assert verdict.evaluate_stage1(_res(), PR)["overall"] == "pass"


def test_verdict_too_few_placebos_fails_even_if_p_small():
    v = verdict.evaluate_stage1(_res(n_placebos=100), PR)
    assert v["overall"] == "fail" and v["criteria"][2]["status"] == "fail"


def test_verdict_positive_gamma_fails():
    r = _res()
    r["estimates"]["surprise"]["gamma"] = 0.01
    assert verdict.evaluate_stage1(r, PR)["criteria"][0]["status"] == "fail"


def test_verdict_missing_offset_scan_is_incomplete_not_pass():
    v = verdict.evaluate_stage1(_res(offset_scan=None), PR)
    assert v["overall"] == "incomplete"


def test_verdict_offset_peak_outside_tolerance_fails():
    scan = [{"delta_s": -1800, "loglik": -1.0}, {"delta_s": 0, "loglik": -9.0}]
    assert verdict.evaluate_stage1(_res(offset_scan=scan), PR)["criteria"][4]["status"] == "fail"


# ------------------------------------------------------------------ checks
def test_holm_matches_reference():
    assert checks.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_size_calibration_fails_on_oversized_null_and_ignores_other_dgp(tmp_path, monkeypatch):
    from autoresearch import core
    monkeypatch.setattr(checks, "code_hash", lambda: "H")
    led = Ledger(tmp_path / "l.jsonl")
    bad = {"is_null": True, "level_sd": 0.08, "n_sims": 16, "nominal": 0.05, "reject_robust": 14, "reject_cluster": 2,
           "binom_p_robust_above_nominal": 1e-16}
    led.append("experiment", experiment="mc_calibration", code_hash="H", mc=bad)
    # a later, idealised (no day-shock) run that looks fine must NOT overwrite the realistic evidence
    led.append("experiment", experiment="mc_calibration", code_hash="H",
               mc={**bad, "level_sd": 0.0, "reject_robust": 1, "binom_p_robust_above_nominal": 0.6})
    r = checks.size_calibration({}, {"pr": PR, "ledger": led})
    assert r["status"] == "fail"


def test_size_calibration_concern_when_no_evidence_for_current_code(tmp_path, monkeypatch):
    monkeypatch.setattr(checks, "code_hash", lambda: "NEW")
    led = Ledger(tmp_path / "l.jsonl")
    led.append("experiment", experiment="mc_calibration", code_hash="OLD", mc={"is_null": True, "level_sd": 0.08})
    assert checks.size_calibration({}, {"pr": PR, "ledger": led})["status"] == "concern"


def test_se_validity_flags_robust_only_significance():
    res = {"wald_robust": {"p": 1e-8}, "wald_cluster": {"p": 0.2},
           "estimates": {k: {"long_run": {"se_robust": 0.03, "se_cluster": 0.07}} for k in ("scheduled", "surprise")}}
    assert checks.se_validity(res, {"pr": PR})["status"] == "concern"


# ------------------------------------------------------------------ vetting rules
F_POS = {"claims_effect": True}
F_NEG = {"claims_effect": False}
OK = {"lens": "x", "verdict": "pass", "issues": []}
BLK = {"lens": "y", "verdict": "block", "issues": [{"severity": "blocker", "claim": "leak"}]}
CON = {"lens": "z", "verdict": "concern", "issues": [{"severity": "minor", "claim": "thin"}]}
FAIL = {"id": "size_calibration", "status": "fail", "detail": "oversized"}


def test_vet_cannot_verify_without_llm():
    assert vet(F_NEG, [], [], llm_ran=False, final=True)["status"] == "DETERMINISTIC_ONLY"


def test_vet_failed_check_rejects_positive_claim_but_only_caveats_a_null():
    assert vet(F_POS, [FAIL], [OK], True, False)["status"] == "REJECTED"
    assert vet(F_NEG, [FAIL], [OK], True, False)["status"] == "VERIFIED_WITH_CAVEATS"


def test_vet_block_triggers_revision_then_rejection_or_human():
    assert vet(F_NEG, [], [BLK, OK], True, final=False)["status"] == "NEEDS_REVISION"
    assert vet(F_NEG, [], [BLK, BLK], True, final=True)["status"] == "REJECTED"
    assert vet(F_NEG, [], [BLK, OK], True, final=True)["status"] == "NEEDS_HUMAN"


def test_vet_clean_is_verified_and_concerns_become_caveats():
    assert vet(F_NEG, [], [OK, OK], True, True)["status"] == "VERIFIED"
    assert vet(F_NEG, [], [OK, CON], True, True)["status"] == "VERIFIED_WITH_CAVEATS"


# ------------------------------------------------------------------ repro runner (code-audit lane)
def test_repro_requires_assertionerror(tmp_path):
    assert run_repro("assert 1 == 2, 'bug'", tmp_path, 0)["verified"] is True
    assert run_repro("assert 1 == 1", tmp_path, 1)["verified"] is False                 # exits 0: no bug shown
    assert run_repro("import nonexistent_module_xyz", tmp_path, 2)["verified"] is False   # crash != bug


# ------------------------------------------------------------------ normalization handles both runners
def test_normalize_handles_run_layer1_and_run_real_shapes():
    l1 = {"args": {"placebos": 50, "half_life": 1800.0}, "exp_fit": [{"index": "gamma_z_scheduled", "estimate": -0.03, "se_robust": 0.002},
          {"index": "gamma_z_surprise", "estimate": -0.01, "se_robust": 0.002}],
          "long_run": {"scheduled": {"effect": -0.5, "se": 0.03}, "surprise": {"effect": -0.2, "se": 0.04}},
          "long_run_cluster_day": {"scheduled": {"effect": -0.5, "se": 0.07}, "surprise": {"effect": -0.2, "se": 0.09}},
          "placebo": {"scheduled": {"observed": -0.03, "placebo_mean": 0.0, "placebo_sd": 0.01, "randomization_p": 0.04},
                      "surprise": {"observed": -0.01, "placebo_mean": 0.0, "placebo_sd": 0.01, "randomization_p": 0.2}},
          "wald_robust": {"p": 1e-9}, "event_counts": {"scheduled|intraday=False": 10, "surprise|intraday=True": 7}}
    real = {"args": {"placebos": 200, "placebo_mode": "nuisance"},
            "exp_fit": [{"index": "gamma_z_scheduled", "estimate": -0.03, "se_robust": 0.002, "se_cluster_day": 0.006},
                        {"index": "gamma_z_surprise", "estimate": -0.01, "se_robust": 0.002, "se_cluster_day": 0.006}],
            "long_run": {k: {"robust": {"effect": -0.5, "se": 0.03}, "cluster": {"effect": -0.5, "se": 0.07}} for k in ("scheduled", "surprise")},
            "placebo": {k: {"observed": -0.03, "placebo_mean": 0.0, "placebo_sd": 0.01, "p_two_sided": 0.01} for k in ("scheduled", "surprise")},
            "wald_robust": {"p": 1e-9}, "event_counts": [{"kind": "scheduled", "intraday": False, "n": 12}]}
    a, b = normalize_layer1(l1, "t", "real"), normalize_layer1(real, "t", "real")
    assert a["estimates"]["scheduled"]["long_run"]["se_cluster"] == 0.07 == b["estimates"]["scheduled"]["long_run"]["se_cluster"]
    assert a["placebo"]["scheduled"]["p"] == 0.04 and b["placebo"]["scheduled"]["p"] == 0.01
    assert a["event_counts"] == {"scheduled": 10, "surprise": 7} and b["event_counts"] == {"scheduled": 12}
    assert b["spec"]["placebo_mode"] == "nuisance"
