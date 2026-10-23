import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decompiler_engine.models import Instruction
from decompiler_engine.pseudocode import PseudocodeLifter

def mk(mnemonic, operands="", target=None, ref=None, flow=None):
    defaults = {"call":"call", "jmp":"jump", "ret":"return", "jle":"conditional", "jne":"conditional"}
    return Instruction(0x1000, 2, "90 90", mnemonic, operands, ".text", target, ref, flow or defaults.get(mnemonic, "normal"))

def test_basic_arithmetic_lifting():
    lifter = PseudocodeLifter(64)
    assert lifter.lift(mk("add", "eax, edx"), set()) == "v_rax = v_rax + v_rdx;"
    assert lifter.lift(mk("imul", "eax, eax"), set()) == "v_rax = v_rax * v_rax;"
    assert lifter.lift(mk("xor", "eax, eax"), set()) == "v_rax = 0;"

def test_conditional_branch_keeps_target():
    lifter = PseudocodeLifter(64)
    ins = mk("jle", "0x1010", target=0x1010)
    assert lifter.lift(ins, {0x1010}) == "if (condition_less_or_equal) goto loc_1010;"

def test_string_address_in_lea():
    lifter = PseudocodeLifter(64, strings={0x2222: "hello"})
    ins = mk("lea", "rcx, [rip + 0x10]", ref=0x2222)
    assert '"hello"' in lifter.lift(ins, set())

def test_memory_operand_conversion():
    lifter = PseudocodeLifter(64)
    result = lifter.lift(mk("mov", "dword ptr [rbp - 4], eax"), set())
    assert result == "local_4 = v_rax;"


def test_rsp_relative_stack_slots_are_named_locals_or_arguments():
    lifter = PseudocodeLifter(64)
    local = lifter.lift(mk("mov", "dword ptr [rsp - 8], eax"), set())
    argument = lifter.lift(mk("mov", "eax, dword ptr [rsp + 20]"), set())
    assert local == "local_8 = v_rax;"
    assert argument == "v_rax = stack_arg_14;"

def test_call_uses_symbolic_function_name():
    lifter = PseudocodeLifter(64, function_names={0x2000: "target_function"})
    result = lifter.lift(mk("call", "0x2000", target=0x2000, flow="call"), set())
    assert result.startswith("target_function(")


def test_register_call_after_iat_load_uses_import_symbol():
    from decompiler_engine.models import BasicBlock
    load = Instruction(
        0x2400, 1, "90", "mov", "rax, qword ptr [rip + 0x10]", ".text",
        reference_target=0x5000,
    )
    call = Instruction(0x2401, 1, "90", "call", "rax", ".text", flow="call")
    ret = Instruction(0x2402, 1, "90", "ret", "", ".text", flow="return")
    block = BasicBlock(0x2400, 0x2403, [load, call, ret], [], ".text")
    lifter = PseudocodeLifter(64, imports={0x5000: "KERNEL32.dll!GetLastError"})
    lifter.infer_function_signatures({0x2400: [block]})
    output = lifter.function_text("caller", 0x2400, [block])
    assert "GetLastError(); /* import: KERNEL32.dll!GetLastError */" in output


def test_instruction_serializes_jump_table_and_indirect_call_metadata():
    instruction = Instruction(
        0x1000, 2, "90", "jmp", "rax", ".text", flow="jump",
        indirect_targets=[0x1010, 0x1020], jump_table_entry_size=4,
        jump_table_base=0x3000, indirect_call_targets=[0x4000],
        jump_table_entries_observed=4, jump_table_index_register="rcx",
        jump_table_case_count=3, jump_table_bound_branch="ja",
        jump_table_bound_instruction=0x0FF0, jump_table_index_bias=10,
        jump_table_default_target=0x2000, jump_table_default_entry_indices=[2],
        jump_table_slots=[0x1010, None, 0x1020],
        resolved_indirect_jump_target=0x4000,
    )
    data = instruction.to_dict()
    assert data["indirect_targets"] == ["0x1010", "0x1020"]
    assert data["jump_table_entry_size"] == 4
    assert data["jump_table_base"] == "0x3000"
    assert data["jump_table_entries_observed"] == 4
    assert data["jump_table_index_register"] == "rcx"
    assert data["jump_table_case_count"] == 3
    assert data["jump_table_bound_branch"] == "ja"
    assert data["jump_table_bound_instruction"] == "0xff0"
    assert data["jump_table_index_bias"] == 10
    assert data["jump_table_default_target"] == "0x2000"
    assert data["jump_table_default_entry_indices"] == [2]
    assert data["jump_table_slots"] == ["0x1010", None, "0x1020"]
    assert data["resolved_indirect_jump_target"] == "0x4000"
    assert data["indirect_call_targets"] == ["0x4000"]


def test_guarded_jump_table_lifts_to_biased_switch_with_opaque_holes():
    lifter = PseudocodeLifter(64)
    instruction = Instruction(
        0x2000, 6, "FF 24", "jmp", "qword ptr [rax + rcx*8]", ".text", flow="jump",
        indirect_targets=[0x2010, 0x2020], jump_table_index_register="rcx",
        jump_table_case_count=3, jump_table_index_bias=10,
        jump_table_default_target=0x2030,
        jump_table_slots=[0x2010, None, 0x2020],
    )
    lifted = lifter.lift(instruction, {0x2010, 0x2020, 0x2030})
    assert lifted == (
        "switch ((v_rcx + 10)) { case 10: goto loc_2010; "
        "case 11: goto_indirect(qword ptr [rax + rcx*8]); /* unresolved table slot */ "
        "case 12: goto loc_2020; default: goto loc_2030; }"
    )


def test_guarded_jump_table_with_truncated_slots_stays_indirect():
    lifter = PseudocodeLifter(64)
    instruction = Instruction(
        0x2100, 6, "FF 24", "jmp", "qword ptr [rax + rcx*8]", ".text", flow="jump",
        indirect_targets=[0x2110], jump_table_index_register="rcx",
        jump_table_case_count=4, jump_table_index_bias=0,
        jump_table_default_target=0x2130, jump_table_slots=[0x2110, None],
    )
    assert lifter.lift(instruction, {0x2110, 0x2130}) == (
        "goto_indirect(qword ptr [rax + rcx*8]);"
    )


def test_resolved_indirect_tail_jump_lifts_to_recovered_label():
    lifter = PseudocodeLifter(64)
    instruction = Instruction(
        0x2200, 2, "FF E0", "jmp", "rax", ".text", flow="jump",
        resolved_indirect_jump_target=0x2210,
    )
    assert lifter.lift(instruction, {0x2210}) == (
        "goto loc_2210; /* resolved indirect tail-jump target */"
    )


def test_cmp_then_conditional_recovers_comparison_expression():
    lifter = PseudocodeLifter(64)
    cmp_ins = mk("cmp", "dword ptr [rbp - 8], 0x1e")
    branch = mk("jle", "0x1010", target=0x1010, flow="conditional")
    expr = lifter._condition_expression(branch, cmp_ins)
    assert expr == "(local_8 <= 0x1e)"
    assert lifter.lift(branch, {0x1010}, expr) == "if ((local_8 <= 0x1e)) goto loc_1010;"
