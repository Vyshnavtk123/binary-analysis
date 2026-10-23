import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decompiler_engine.disassembly import Disassembler
from decompiler_engine.models import Instruction


def ins(address, mnemonic, operands="", *, target=None, flow="normal"):
    return Instruction(address, 2, "90 90", mnemonic, operands, ".text", target, None, flow)


def test_jump_table_inclusive_unsigned_guard_recovers_index_and_count():
    instructions = [
        ins(0x1000, "cmp", "ecx, 5"),
        ins(0x1002, "ja", "0x1100", target=0x1100, flow="conditional"),
        ins(0x1004, "jmp", "qword ptr [rip + rcx*8 + 0x20]", flow="jump"),
    ]
    bound = Disassembler._infer_jump_table_bound(instructions, 2)
    assert bound == {
        "index_register": "rcx",
        "case_count": 6,
        "index_bias": 0,
        "branch": "ja",
        "default_target": 0x1100,
        "compare_address": 0x1000,
    }


def test_jump_table_exclusive_unsigned_guard_handles_register_aliases():
    instructions = [
        ins(0x2000, "cmp", "r9d, 0x4"),
        ins(0x2002, "jae", "0x2100", target=0x2100, flow="conditional"),
        ins(0x2004, "jmp", "dword ptr [rip + r9*4 + 0x40]", flow="jump"),
    ]
    bound = Disassembler._infer_jump_table_bound(instructions, 2)
    assert bound is not None
    assert bound["index_register"] == "r9"
    assert bound["case_count"] == 4
    assert bound["index_bias"] == 0
    assert bound["default_target"] == 0x2100


def test_jump_table_signed_upper_guards_require_matching_nonnegative_guard():
    inclusive = [
        ins(0x2200, "test", "ecx, ecx"),
        ins(0x2202, "js", "0x2300", target=0x2300, flow="conditional"),
        ins(0x2204, "cmp", "ecx, 5"),
        ins(0x2206, "jg", "0x2300", target=0x2300, flow="conditional"),
        ins(0x2208, "jmp", "qword ptr [rip + rcx*8 + 0x40]", flow="jump"),
    ]
    bound = Disassembler._infer_jump_table_bound(inclusive, 4)
    assert bound is not None
    assert bound["case_count"] == 6
    assert bound["branch"] == "jg"
    assert bound["default_target"] == 0x2300

    exclusive = [
        ins(0x2400, "cmp", "ecx, 0"),
        ins(0x2402, "jl", "0x2500", target=0x2500, flow="conditional"),
        ins(0x2404, "cmp", "ecx, 5"),
        ins(0x2406, "jge", "0x2500", target=0x2500, flow="conditional"),
        ins(0x2408, "jmp", "qword ptr [rip + rcx*8 + 0x40]", flow="jump"),
    ]
    bound = Disassembler._infer_jump_table_bound(exclusive, 4)
    assert bound is not None
    assert bound["case_count"] == 5
    assert bound["branch"] == "jge"

    missing_lower_guard = [
        ins(0x2600, "cmp", "ecx, 5"),
        ins(0x2602, "jg", "0x2700", target=0x2700, flow="conditional"),
        ins(0x2604, "jmp", "qword ptr [rip + rcx*8 + 0x40]", flow="jump"),
    ]
    assert Disassembler._infer_jump_table_bound(missing_lower_guard, 2) is None


def test_jump_table_default_case_entries_are_reported_by_index():
    targets = [0x1010, 0x1030, 0x1010, 0x1040, 0x1030]
    assert Disassembler._jump_table_default_entry_indices(targets, 0x1030) == [1, 4]
    assert Disassembler._jump_table_default_entry_indices(
        [0x1010, None, 0x1030, 0x1040, 0x1030], 0x1030
    ) == [2, 4]
    assert Disassembler._jump_table_default_entry_indices(targets, None) == []


def test_jump_table_recovers_subtract_and_lea_case_value_biases():
    subtract = [
        ins(0x2500, "sub", "ecx, 10"),
        ins(0x2502, "cmp", "ecx, 5"),
        ins(0x2504, "ja", "0x2600", target=0x2600, flow="conditional"),
        ins(0x2506, "jmp", "qword ptr [rip + rcx*8 + 0x40]", flow="jump"),
    ]
    bound = Disassembler._infer_jump_table_bound(subtract, 3)
    assert bound == {
        "index_register": "rcx", "case_count": 6, "index_bias": 10,
        "branch": "ja", "default_target": 0x2600, "compare_address": 0x2502,
    }

    lea = [
        ins(0x2700, "lea", "r9d, [r9d - 0x20]"),
        ins(0x2702, "cmp", "r9d, 3"),
        ins(0x2704, "jae", "0x2800", target=0x2800, flow="conditional"),
        ins(0x2706, "jmp", "dword ptr [rip + r9*4 + 0x40]", flow="jump"),
    ]
    bound = Disassembler._infer_jump_table_bound(lea, 3)
    assert bound is not None
    assert bound["case_count"] == 3
    assert bound["index_bias"] == 0x20


def test_jump_table_bound_is_rejected_when_index_or_fallthrough_does_not_match():
    mismatched = [
        ins(0x3000, "cmp", "edx, 5"),
        ins(0x3002, "ja", "0x3100", target=0x3100, flow="conditional"),
        ins(0x3004, "jmp", "qword ptr [rip + rcx*8]", flow="jump"),
    ]
    assert Disassembler._infer_jump_table_bound(mismatched, 2) is None

    branch_in_fallthrough = [
        ins(0x4000, "cmp", "ecx, 5"),
        ins(0x4002, "ja", "0x4004", target=0x4004, flow="conditional"),
        ins(0x4004, "jmp", "qword ptr [rip + rcx*8]", flow="jump"),
    ]
    assert Disassembler._infer_jump_table_bound(branch_in_fallthrough, 2) is None
