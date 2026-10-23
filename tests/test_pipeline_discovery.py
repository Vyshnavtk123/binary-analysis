import json
import sys
import tempfile
from threading import Event
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decompiler_engine.models import Instruction
from decompiler_engine.disassembly import Disassembler
from decompiler_engine.pipeline import DecompilerPipeline


def test_pipeline_promotes_indirect_call_and_unwind_handler_targets():
    instructions = [
        Instruction(0x1000, 1, "90", "mov", "rax, 0x2000", ".text"),
        Instruction(0x1001, 1, "90", "call", "rax", ".text", flow="call"),
        Instruction(0x1002, 1, "90", "mov", "rax, qword ptr [rip + 0x10]", ".text", reference_target=0x5000),
        Instruction(0x1003, 1, "90", "call", "rax", ".text", flow="call"),
        Instruction(0x1004, 1, "90", "mov", "rbx, qword ptr [rip + 0x20]", ".text", reference_target=0x6000),
        Instruction(0x1005, 1, "90", "call", "qword ptr [rbx + 8]", ".text", flow="call"),
        Instruction(0x1006, 1, "90", "mov", "rcx, 0x4000", ".text"),
        Instruction(0x1007, 1, "90", "call", "0x3000", ".text", direct_target=0x3000, flow="call"),
        Instruction(0x1008, 1, "90", "lea", "rdx, [rip + 0x20]", ".text", reference_target=0x6100),
        Instruction(0x1009, 1, "90", "mov", "rsi, qword ptr [rdx]", ".text"),
        Instruction(0x100A, 1, "90", "call", "qword ptr [rsi + 0x10]", ".text", flow="call"),
        Instruction(0x100B, 1, "90", "mov", "rax, 0x5000", ".text"),
        Instruction(0x100C, 1, "90", "jmp", "rax", ".text", flow="jump"),
        Instruction(0x2000, 1, "90", "mov", "eax, ecx", ".text"),
        Instruction(0x2001, 1, "90", "ret", "", ".text", flow="return"),
        Instruction(0x3000, 1, "90", "mov", "eax, 1", ".text"),
        Instruction(0x3001, 1, "90", "ret", "", ".text", flow="return"),
        Instruction(0x4000, 1, "90", "ret", "", ".text", flow="return"),
        Instruction(0x5000, 1, "90", "ret", "", ".text", flow="return"),
    ]
    image = SimpleNamespace(
        bits=64,
        architecture="x86-64",
        image_base=0x1000,
        entry_point=0x1000,
        sections=[SimpleNamespace(start=0x1000, end=0x8000, backed_size=0x7000, raw_size=0x7000, virtual_size=0x7000,
                                  executable=True, readable=True, writable=False, name=".text")],
        imports={0x5000: "KERNEL32.dll!GetLastError"},
        exports={},
        runtime_functions=[{
            "start": 0x1000, "end": 0x1010, "unwind_info": 0x8000,
            "handler_va": 0x3000, "handler_rva": 0x2000,
            "unwind_flag_names": ["EHANDLER"],
            "chained_start": 0x2000, "chained_end": 0x2002,
            "chained_unwind_info": 0x8100,
        }],
        runtime_function_starts=set(),
        exception_handler_starts={0x3000},
        scan_pointer_slots=lambda max_slots: (
            {0x6000: 0x7000, 0x7008: 0x3000, 0x6100: 0x7100, 0x7110: 0x4000},
            {"slots_scanned": 4, "pointer_slots_found": 4,
             "slot_budget": max_slots, "budget_exhausted": False},
        ),
        scan_strings=lambda *args, **kwargs: [],
        close=lambda: None,
    )
    decoder = SimpleNamespace(run=lambda: instructions, warnings=[])

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_file = root / "sample.exe"
        input_file.write_bytes(b"mock PE")
        output_dir = root / "analysis"
        with patch("decompiler_engine.pipeline.PEImage", return_value=image), patch(
            "decompiler_engine.pipeline.Disassembler", return_value=decoder
        ):
            summary = DecompilerPipeline(
                str(input_file), str(output_dir),
                write_assembly=False, write_ir=False,
            ).run()

        functions = json.loads((output_dir / "functions.json").read_text(encoding="utf-8"))
        report = json.loads((output_dir / "program.json").read_text(encoding="utf-8"))
        target = next(row for row in functions if row["address"] == "0x2000")
        handler = next(row for row in functions if row["address"] == "0x3000")
        callback = next(row for row in functions if row["address"] == "0x4000")
        tail_target = next(row for row in functions if row["address"] == "0x5000")
        caller_function = next(row for row in functions if row["address"] == "0x1000")
        assert "resolved indirect call target (value flow)" in target["reason"]
        assert "Windows exception/unwind handler metadata" in handler["reason"]
        assert "callback candidate" in callback["reason"]
        assert "resolved indirect tail-jump target" in tail_target["reason"]
        assert "0x5000" in caller_function["calls"]
        assert summary["function_candidates"] == 5
        assert summary["indirect_call_discovery"]["targets_resolved"] == ["0x2000", "0x3000", "0x4000"]
        assert summary["indirect_call_discovery"]["tail_jump_targets"] == ["0x5000"]
        assert summary["indirect_call_discovery"]["callback_targets"] == ["0x4000"]
        assert summary["static_pointer_scan"]["pointer_slots_found"] == 4
        assert summary["windows_x64_exception_handler_candidates"] == 1
        assert report["exceptional_edges"] == [{
            "kind": "windows_x64_unwind_handler_candidate",
            "source_function": "0x1000",
            "protected_range": {"start": "0x1000", "end": "0x1010"},
            "target_handler": "0x3000",
            "target_function_recovered": True,
            "unwind_info": "0x8000",
            "flags": ["EHANDLER"],
            "control_flow_proven": False,
        }]
        assert report["chained_unwind_relationships"] == [{
            "kind": "windows_x64_chained_runtime_function",
            "source_function": "0x1000",
            "target_function": "0x2000",
            "target_range": {"start": "0x2000", "end": "0x2002"},
            "target_function_recovered": True,
            "unwind_info": "0x8100",
        }]
        assert report["indirect_control_transfers"] == [{
            "source": "0x1001",
            "kind": "indirect_call",
            "reference": None,
            "import": None,
            "targets": ["0x2000"],
            "jump_table": None,
            "resolved_target": None,
        }, {
            "source": "0x1003",
            "kind": "indirect_call",
            "reference": None,
            "import": "KERNEL32.dll!GetLastError",
            "targets": [],
            "jump_table": None,
            "resolved_target": None,
        }, {
            "source": "0x1005",
            "kind": "indirect_call",
            "reference": None,
            "import": None,
            "targets": ["0x3000"],
            "jump_table": None,
            "resolved_target": None,
        }, {
            "source": "0x100a",
            "kind": "indirect_call",
            "reference": None,
            "import": None,
            "targets": ["0x4000"],
            "jump_table": None,
            "resolved_target": None,
        }, {
            "source": "0x100c",
            "kind": "indirect_tail_jump",
            "reference": None,
            "import": None,
            "targets": ["0x5000"],
            "jump_table": None,
            "resolved_target": "0x5000",
        }]
        flow_rows = [json.loads(line) for line in (output_dir / "value_flow.jsonl").read_text(encoding="utf-8").splitlines()]
        import_call = next(row for row in flow_rows if row["address"] == "0x1003")
        assert import_call["resolved_call_import"] == "KERNEL32.dll!GetLastError"
        virtual_call = next(row for row in flow_rows if row["address"] == "0x100a")
        assert virtual_call["resolved_call_provenance"] == "static_pointer_slot"
        assert virtual_call["resolved_call_target"] == "0x4000"
        decompiler_rows = [json.loads(line) for line in (output_dir / "decompiler.jsonl").read_text(encoding="utf-8").splitlines()]
        caller = next(row for row in decompiler_rows if row["address"] == "0x1000")
        assert "sub_3000(" in caller["code"]
        assert "goto address_5000; /* resolved indirect tail-jump target */" in caller["code"]

        cancelled_dir = root / "cancelled"
        cancel_event = Event()
        progress_events = []

        def cancel_after_one_function(event):
            progress_events.append(event)
            if event["stage"] == "function_index" and event["completed"] >= 1:
                cancel_event.set()

        with patch("decompiler_engine.pipeline.PEImage", return_value=image), patch(
            "decompiler_engine.pipeline.Disassembler", return_value=decoder
        ):
            cancelled_summary = DecompilerPipeline(
                str(input_file), str(cancelled_dir), write_assembly=False, write_ir=False,
                progress_callback=cancel_after_one_function, cancel_check=cancel_event.is_set,
            ).run()
        cancelled_functions = json.loads((cancelled_dir / "functions.json").read_text(encoding="utf-8"))
        assert cancelled_summary["analysis_complete"] is False
        assert cancelled_summary["cancelled"] is True
        assert cancelled_summary["cancelled_stage"] == "function_index"
        assert len(cancelled_functions) == 1
        assert any(event["stage"] == "function_index" for event in progress_events)
        assert any("Analysis cancelled" in warning for warning in cancelled_summary["warnings"])


def test_metadata_symbol_without_call_path_has_unknown_application_reachability():
    instructions = [
        Instruction(0x1000, 1, "C3", "ret", "", ".text", flow="return"),
        Instruction(0x1100, 1, "C3", "ret", "", ".text", flow="return"),
    ]
    image = SimpleNamespace(
        bits=64, architecture="x86-64", image_base=0x1000, entry_point=0x1000,
        sections=[SimpleNamespace(start=0x1000, end=0x1200, backed_size=0x200, raw_size=0x200, virtual_size=0x200,
                                  executable=True, readable=True, writable=False, name=".text")],
        imports={}, exports={0x1100: "orphan_export"}, runtime_functions=[],
        runtime_function_starts=set(), exception_handler_starts=set(), go_functions={},
        scan_pointer_slots=lambda max_slots: ({}, {
            "slots_scanned": 0, "pointer_slots_found": 0,
            "slot_budget": max_slots, "budget_exhausted": False,
        }),
        scan_strings=lambda *args, **kwargs: [], close=lambda: None,
    )
    decoder = SimpleNamespace(run=lambda: instructions, warnings=[])
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_file = root / "metadata_only.exe"
        input_file.write_bytes(b"mock PE")
        output_dir = root / "analysis"
        with patch("decompiler_engine.pipeline.PEImage", return_value=image), patch(
            "decompiler_engine.pipeline.Disassembler", return_value=decoder
        ):
            DecompilerPipeline(
                str(input_file), str(output_dir), write_assembly=False, write_ir=False,
            ).run()
        functions = json.loads((output_dir / "functions.json").read_text(encoding="utf-8"))
    export = next(row for row in functions if row["address"] == "0x1100")
    assert export["analysis_states"]["symbol_present"] is True
    assert export["analysis_states"]["reachable_from_application_root"] is None
    assert "unknown" in export["analysis_states"]["states"]


def test_dataflow_target_is_recursively_decoded_before_call_edge_is_added():
    # The call target is present in linear-sweep candidates but is not reachable
    # from the entry seed until value flow resolves RAX at the indirect call.
    code = bytearray(b"\x90" * 0x1001)
    code[0:7] = bytes.fromhex("48 C7 C0 00 20 00 00")  # mov rax, 0x2000
    code[7:9] = bytes.fromhex("FF D0")  # call rax
    code[9] = 0xC3
    code[0x1000] = 0xC3  # hidden function at VA 0x2000
    section = SimpleNamespace(
        start=0x1000, end=0x2001, backed_size=len(code), raw_size=len(code),
        virtual_size=len(code), executable=True, readable=True, writable=False, name=".text",
    )
    image = SimpleNamespace(
        bits=64, architecture="x86-64", image_base=0x1000, entry_point=0x1000,
        sections=[section], imports={}, exports={}, runtime_functions=[],
        runtime_function_starts=set(), exception_handler_starts=set(), go_functions={},
        scan_pointer_slots=lambda max_slots: ({}, {
            "slots_scanned": 0, "pointer_slots_found": 0,
            "slot_budget": max_slots, "budget_exhausted": False,
        }),
        scan_strings=lambda *args, **kwargs: [],
        section_bytes=lambda _section: bytes(code), close=lambda: None,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_file = root / "indirect_target.exe"
        input_file.write_bytes(b"mock PE")
        output_dir = root / "analysis"
        with patch("decompiler_engine.pipeline.PEImage", return_value=image), patch(
            "decompiler_engine.pipeline.Disassembler", side_effect=lambda pe, cap: Disassembler(pe, cap)
        ):
            summary = DecompilerPipeline(
                str(input_file), str(output_dir), write_assembly=False, write_ir=True,
                max_functions=20, max_pseudocode_functions=20,
                max_dataflow_functions=20, max_dataflow_instructions=2000,
            ).run()
        functions = json.loads((output_dir / "functions.json").read_text(encoding="utf-8"))
        assembly_ir = json.loads((output_dir / "ir.jsonl").read_text(encoding="utf-8").splitlines()[0])
    target = next(row for row in functions if row["address"] == "0x2000")
    assert "resolved indirect call target (value flow)" in target["reason"]
    assert target["analysis_states"]["decoded_from_seed"] is True
    assert summary["indirect_call_discovery"]["newly_decoded_targets"] == ["0x2000"]
    assert assembly_ir["discovery_source"] == "recursive_control_flow"


def test_x86_esp_slot_callback_becomes_function_candidate():
    instructions = [
        Instruction(0x1000, 1, "90", "mov", "eax, 0x2000", ".text"),
        Instruction(0x1001, 1, "90", "sub", "esp, 4", ".text"),
        Instruction(0x1002, 1, "90", "mov", "dword ptr [esp], eax", ".text"),
        Instruction(0x1003, 1, "90", "jmp", "0x1004", ".text", direct_target=0x1004, flow="jump"),
        Instruction(0x1004, 1, "90", "call", "0x3000", ".text", direct_target=0x3000, flow="call"),
        Instruction(0x1005, 1, "90", "add", "esp, 4", ".text"),
        Instruction(0x1006, 1, "90", "ret", "", ".text", flow="return"),
        Instruction(0x2000, 1, "90", "ret", "", ".text", flow="return"),
        Instruction(0x3000, 1, "90", "ret", "", ".text", flow="return"),
    ]
    image = SimpleNamespace(
        bits=32, architecture="x86", image_base=0x1000, entry_point=0x1000,
        sections=[SimpleNamespace(start=0x1000, end=0x4000, backed_size=0x3000, raw_size=0x3000, virtual_size=0x3000,
                                  executable=True, readable=True, writable=False, name=".text")], imports={}, exports={}, runtime_functions=[],
        runtime_function_starts=set(), exception_handler_starts=set(),
        scan_pointer_slots=lambda max_slots: ({}, {
            "slots_scanned": 0, "pointer_slots_found": 0,
            "slot_budget": max_slots, "budget_exhausted": False,
        }),
        scan_strings=lambda *args, **kwargs: [], close=lambda: None,
    )
    decoder = SimpleNamespace(run=lambda: instructions, warnings=[])

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_file = root / "mock_x86.exe"
        input_file.write_bytes(b"mock PE")
        output_dir = root / "analysis"
        with patch("decompiler_engine.pipeline.PEImage", return_value=image), patch(
            "decompiler_engine.pipeline.Disassembler", return_value=decoder
        ):
            summary = DecompilerPipeline(
                str(input_file), str(output_dir), write_assembly=False, write_ir=False,
            ).run()

        functions = json.loads((output_dir / "functions.json").read_text(encoding="utf-8"))
        callback = next(row for row in functions if row["address"] == "0x2000")
        assert "callback candidate" in callback["reason"]
        assert summary["indirect_call_discovery"]["callback_targets"] == ["0x2000"]


def test_complete_prologue_based_jump_table_seeds_function_candidates():
    instructions = [
        Instruction(0x1000, 2, "FF E0", "jmp", "rax", ".text", flow="jump",
                    indirect_targets=[0x2000, 0x3000], jump_table_case_count=2,
                    jump_table_slots=[0x2000, 0x3000]),
        Instruction(0x2000, 1, "55", "push", "rbp", ".text"),
        Instruction(0x2001, 3, "48 89 E5", "mov", "rbp, rsp", ".text"),
        Instruction(0x2004, 1, "C3", "ret", "", ".text", flow="return"),
        Instruction(0x3000, 1, "55", "push", "rbp", ".text"),
        Instruction(0x3001, 3, "48 89 E5", "mov", "rbp, rsp", ".text"),
        Instruction(0x3004, 1, "C3", "ret", "", ".text", flow="return"),
    ]
    image = SimpleNamespace(
        bits=64, architecture="x86-64", image_base=0x1000, entry_point=0x1000,
        sections=[], imports={}, exports={}, runtime_functions=[],
        runtime_function_starts=set(), exception_handler_starts=set(),
        scan_pointer_slots=lambda max_slots: ({}, {
            "slots_scanned": 0, "pointer_slots_found": 0,
            "slot_budget": max_slots, "budget_exhausted": False,
        }),
        scan_strings=lambda *args, **kwargs: [], close=lambda: None,
    )
    decoder = SimpleNamespace(run=lambda: instructions, warnings=[])

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_file = root / "mock_switch_dispatch.exe"
        input_file.write_bytes(b"mock PE")
        output_dir = root / "analysis"
        with patch("decompiler_engine.pipeline.PEImage", return_value=image), patch(
            "decompiler_engine.pipeline.Disassembler", return_value=decoder
        ):
            summary = DecompilerPipeline(
                str(input_file), str(output_dir), write_assembly=False, write_ir=False,
            ).run()
        functions = json.loads((output_dir / "functions.json").read_text(encoding="utf-8"))
    by_address = {row["address"]: row for row in functions}
    assert "complete jump-table function-dispatch candidate" in by_address["0x2000"]["reason"]
    assert "complete jump-table function-dispatch candidate" in by_address["0x3000"]["reason"]
    assert summary["function_candidates"] == 3
