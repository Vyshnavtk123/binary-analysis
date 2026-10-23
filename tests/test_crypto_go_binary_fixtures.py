"""Optional end-to-end accuracy checks for real Go/Windows PE fixtures.

Run with Go, pefile, and Capstone installed. The test records known-positive,
string-only, dead-code, and indirect-call expectations without executing samples.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


pytest.importorskip("pefile")
pytest.importorskip("capstone")
pytestmark = pytest.mark.skipif(shutil.which("go") is None, reason="Go toolchain is not installed")


FIXTURES = Path(__file__).parent / "fixtures" / "crypto_accuracy"


def scan_fixture(name: str, tmp_path: Path) -> dict:
    executable = tmp_path / f"{name}.exe"
    env = os.environ.copy()
    env.update({"GOOS": "windows", "GOARCH": "amd64", "CGO_ENABLED": "0"})
    subprocess.run(
        ["go", "build", "-trimpath", "-o", str(executable), str(FIXTURES / name / "main.go")],
        check=True, env=env, capture_output=True, text=True,
    )
    from decompiler_engine.pipeline import DecompilerPipeline

    output = tmp_path / f"scan-{name}"
    DecompilerPipeline(
        str(executable), str(output), write_assembly=False, write_ir=False,
        write_ssa=False, write_dataflow=True, max_instructions=1_000_000,
        max_functions=2_000, max_pseudocode_functions=500,
        max_dataflow_instructions=100_000, max_dataflow_functions=400,
    ).run()
    import json
    return json.loads((output / "crypto_findings.json").read_text(encoding="utf-8"))


def algorithm_state(report: dict, name: str) -> dict | None:
    return next((item for item in report["algorithm_findings"] if item["algorithm"] == name), None)


def test_known_aes_binary_has_symbol_and_call_evidence(tmp_path):
    report = scan_fixture("aes", tmp_path)
    aes = algorithm_state(report, "AES")
    assert aes is not None
    assert "symbol_present" in aes["states"]
    assert "call_site_found" in aes["states"]


def test_known_rsa_binary_has_symbol_evidence(tmp_path):
    report = scan_fixture("rsa", tmp_path)
    rsa = algorithm_state(report, "RSA")
    assert rsa is not None
    assert "symbol_present" in rsa["states"]
    assert "call_site_found" in rsa["states"]


def test_string_only_fixture_does_not_claim_symbol_or_call(tmp_path):
    report = scan_fixture("string_only", tmp_path)
    aes = algorithm_state(report, "AES")
    assert aes is not None and aes["states"] == ["string_only"]


def test_unreachable_crypto_fixture_does_not_claim_crypto_symbol(tmp_path):
    report = scan_fixture("unreachable", tmp_path)
    aes = algorithm_state(report, "AES")
    assert aes is None or "symbol_present" not in aes["states"]


def test_indirect_fixture_reports_resolution_or_explicit_unknown(tmp_path):
    report = scan_fixture("indirect", tmp_path)
    graph = report["go_call_graph"]
    assert graph["resolved_indirect_edge_count"] > 0 or graph["unresolved_indirect_call_count"] > 0

