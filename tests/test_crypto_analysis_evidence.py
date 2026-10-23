from types import SimpleNamespace

from decompiler_engine.crypto_analysis import analyze_crypto, build_crypto_cbom


def call(address, *, direct=None, targets=None, operand="rax", provenance=None):
    return SimpleNamespace(
        address=address, flow="call", direct_target=direct,
        indirect_call_targets=targets, resolved_call_provenance=provenance,
        import_name=None, text=f"call {operand}", operands=operand,
    )


def binary_string(text, address=0x5000):
    return SimpleNamespace(address=address, text=text, section=".rdata", encoding="ascii")


def test_crypto_string_only_is_not_promoted_to_symbol_or_call():
    report = analyze_crypto([binary_string("crypto/aes.NewCipher")], {})
    aes = next(item for item in report["algorithm_findings"] if item["algorithm"] == "AES")
    assert aes["status"] == "string_only"
    assert aes["states"] == ["string_only"]
    assert aes["runtime_observed"] is False
    assert report["go_function_symbol_evidence"]["unique_symbols"] == 0
    assert report["go_call_graph"]["edge_count"] == 0


def test_indirect_call_evidence_participates_in_reachability():
    functions = {0x1000: "main.main", 0x1100: "crypto/aes.NewCipher"}
    report = analyze_crypto(
        [], {}, [call(0x1005, targets=[0x1100], provenance="value_flow")],
        go_functions=functions,
    )
    edge = report["go_call_graph"]["edges"][0]
    assert edge["call_kind"] == "resolved_indirect"
    assert edge["resolution_provenance"] == "value_flow"
    aes = next(item for item in report["algorithm_findings"] if item["algorithm"] == "AES")
    assert aes["status"] == "statically_reachable"
    assert "symbol_present" in aes["states"] and "call_site_found" in aes["states"]
    assert any(item["type"] == "indirect_call_site" and item["reachable_edge"] for item in aes["evidence"])


def test_unresolved_indirect_call_is_reported_without_edge():
    report = analyze_crypto(
        [], {}, [call(0x1005)], go_functions={0x1000: "main.main"}
    )
    assert report["go_call_graph"]["edge_count"] == 0
    assert report["go_call_graph"]["unresolved_indirect_call_count"] == 1
    row = report["go_call_graph"]["unresolved_indirect_calls"][0]
    assert row["evidence_type"] == "unresolved_indirect_call"
    assert row["caller"] == "main.main"
    assert row["resolution"] == "unresolved"


def test_memory_indirect_target_has_explicit_provenance():
    report = analyze_crypto(
        [], {}, [call(0x1005, targets=[0x1100], operand="qword ptr [rax]", provenance="static_pointer_slot")],
        go_functions={0x1000: "main.main", 0x1100: "crypto/rsa.SignPSS"},
    )
    edge = report["go_call_graph"]["edges"][0]
    assert edge["call_kind"] == "resolved_indirect"
    assert edge["resolution_provenance"] == "static_pointer_slot"
    assert edge["callee"] == "crypto/rsa.SignPSS"


def test_unmapped_indirect_candidate_is_not_added_as_graph_edge():
    report = analyze_crypto(
        [], {}, [call(0x1005, targets=[0xDEAD], provenance="value_flow")],
        go_functions={0x1000: "main.main"},
        executable_ranges=[(0x1000, 0x2000)],
    )
    assert report["go_call_graph"]["edge_count"] == 0
    unresolved = report["go_call_graph"]["unresolved_indirect_calls"][0]
    assert unresolved["resolution"] == "partially_resolved"
    assert unresolved["unmapped_target_candidates"] == ["0xdead"]


def test_go_crypto_signature_rules_separate_aead_mode_from_cipher_algorithm():
    report = analyze_crypto([], {}, go_functions={
        0x2000: "crypto/cipher.NewGCM",
        0x2100: "crypto/aes.NewCipher",
        0x2200: "crypto/rsa.SignPSS",
    })
    operations = {item["symbol"]: item["operation"] for item in report["crypto_operations_found"]}
    assert operations["crypto/cipher.NewGCM"] == "aead_mode_gcm"
    assert operations["crypto/aes.NewCipher"] == "cipher_initialization"
    assert operations["crypto/rsa.SignPSS"] == "signature_pss"
    assert "AES-GCM" not in report["algorithm_use_states"]


def test_cbom_deduplicates_algorithm_and_preserves_call_relationship_evidence():
    functions = {0x1000: "main.main", 0x1100: "crypto/aes.NewCipher"}
    report = analyze_crypto(
        [binary_string("AES", 0x5000)], {}, [call(0x1005, direct=0x1100)],
        go_functions=functions,
    )
    bom = build_crypto_cbom("sample.exe", report)
    aes = [component for component in bom["components"] if component["name"] == "AES"]
    assert len(aes) == 1
    assert len(aes[0]["evidence"]["occurrences"]) >= 3
    contexts = [item["additionalContext"] for item in aes[0]["evidence"]["occurrences"]]
    assert any('"caller": "main.main"' in context for context in contexts)
    assert bom["metadata"]["tools"]["components"][0]["version"]


def test_cbom_omits_non_enum_families_from_cyclonedx_crypto_properties():
    report = analyze_crypto([], {}, go_functions={
        0x1000: "golang.org/x/crypto/argon2.IDKey",
        0x2000: "golang.org/x/crypto/scrypt.Key",
    })
    bom = build_crypto_cbom("sample.exe", report)
    by_name = {component["name"]: component for component in bom["components"]}
    for algorithm in ("Argon2", "scrypt"):
        props = by_name[algorithm]["cryptoProperties"]["algorithmProperties"]
        assert "algorithmFamily" not in props

