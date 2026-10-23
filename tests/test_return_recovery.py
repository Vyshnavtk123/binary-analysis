from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decompiler_engine.models import BasicBlock, Instruction
from decompiler_engine.pseudocode import PseudocodeLifter


def ins(address, mnemonic, operands="", *, target=None, flow="normal", ref=None,
        import_name=None, resolved_indirect_jump_target=None):
    return Instruction(
        address=address,
        size=1,
        bytes_hex="90",
        mnemonic=mnemonic,
        operands=operands,
        section=".text",
        direct_target=target,
        reference_target=ref,
        flow=flow,
        import_name=import_name,
        resolved_indirect_jump_target=resolved_indirect_jump_target,
    )


def block(start, instructions, successors=()):
    end = instructions[-1].address + 1 if instructions else start
    return BasicBlock(start, end, instructions, list(successors), ".text")


def test_scalar_return_and_one_argument_inferred():
    fn = block(0x1000, [
        ins(0x1000, "mov", "eax, ecx"),
        ins(0x1001, "add", "eax, 2"),
        ins(0x1002, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x1000: "square_like"})
    signatures = lifter.infer_function_signatures({0x1000: [fn]})
    row = signatures[0]
    assert row["return_type"] == "uint32_t"
    assert row["parameter_count"] == 1
    output = lifter.function_text("square_like", 0x1000, [fn])
    assert "uint32_t square_like(uint64_t arg1)" in output
    assert "return (arg1 + 2);" in output


def test_void_function_emits_plain_return():
    fn = block(0x1100, [
        ins(0x1100, "push", "rbp"),
        ins(0x1101, "mov", "rbp, rsp"),
        ins(0x1102, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x1100: "cleanup"})
    lifter.infer_function_signatures({0x1100: [fn]})
    output = lifter.function_text("cleanup", 0x1100, [fn])
    assert "void cleanup(void)" in output
    assert "return;" in output
    assert "return-register/type recovery" not in output


def test_internal_call_arguments_and_return_assignment():
    callee = block(0x2000, [
        ins(0x2000, "mov", "eax, ecx"),
        ins(0x2001, "add", "eax, 2"),
        ins(0x2002, "ret", flow="return"),
    ])
    caller = block(0x3000, [
        ins(0x3000, "mov", "ecx, 6"),
        ins(0x3001, "call", "0x2000", target=0x2000, flow="call"),
        ins(0x3002, "cmp", "eax, 30"),
        ins(0x3003, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x2000: "calc", 0x3000: "caller"})
    lifter.infer_function_signatures({0x2000: [callee], 0x3000: [caller]})
    output = lifter.function_text("caller", 0x3000, [caller])
    assert "v_rax = calc(6);" in output
    assert "uint32_t caller(void)" in output


def test_pointer_return_type_from_import_propagates_through_wrapper():
    lookup = block(0x2380, [
        ins(0x2380, "call", "rax", flow="call", import_name="KERNEL32.dll!GetProcAddress"),
        ins(0x2381, "ret", flow="return"),
    ])
    forward = block(0x2480, [
        ins(0x2480, "call", "0x2380", target=0x2380, flow="call"),
        ins(0x2481, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x2380: "lookup", 0x2480: "forward"})
    signatures = lifter.infer_function_signatures({0x2380: [lookup], 0x2480: [forward]})
    by_address = {row["address"]: row for row in signatures}
    assert by_address["0x2380"]["return_type"] == "uintptr_t"
    assert by_address["0x2480"]["return_type"] == "uintptr_t"
    output = lifter.function_text("forward", 0x2480, [forward])
    assert "uintptr_t forward(void)" in output
    assert "v_rax = lookup(" in output


def test_resolved_indirect_tail_jump_inherits_callee_return_type():
    callee = block(0x2400, [
        ins(0x2400, "mov", "eax, ecx"),
        ins(0x2401, "ret", flow="return"),
    ])
    caller = block(0x2500, [
        ins(0x2500, "jmp", "rax", flow="jump", resolved_indirect_jump_target=0x2400),
    ])
    lifter = PseudocodeLifter(64, function_names={0x2400: "callee", 0x2500: "forward"})
    signatures = lifter.infer_function_signatures({0x2400: [callee], 0x2500: [caller]})
    by_address = {row["address"]: row for row in signatures}
    assert by_address["0x2400"]["return_type"] == "uint32_t"
    assert by_address["0x2500"]["return_type"] == "uint32_t"
    assert by_address["0x2500"]["return_register"] == "tail-call"
    assert by_address["0x2500"]["return_analysis"] == "return type forwarded from a known tail-call target"


def test_fixed_arity_import_arguments():
    fn = block(0x4000, [
        ins(0x4000, "mov", "ecx, 1000"),
        ins(0x4001, "call", "rax", flow="call", import_name="KERNEL32.dll!Sleep"),
        ins(0x4002, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, imports={})
    lifter.infer_function_signatures({0x4000: [fn]})
    output = lifter.function_text("sleep_wrapper", 0x4000, [fn])
    assert "Sleep(1000);" in output


def test_import_pointer_parameter_types_add_casts_to_call_expression():
    fn = block(0x4080, [
        ins(0x4080, "mov", "ecx, 0x1000"),
        ins(0x4081, "mov", "edx, 0x2000"),
        ins(0x4082, "call", "rax", flow="call", import_name="KERNEL32.dll!GetProcAddress"),
        ins(0x4083, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64)
    lifter.infer_function_signatures({0x4080: [fn]})
    output = lifter.function_text("lookup", 0x4080, [fn])
    assert "GetProcAddress(4096, (const char *)(8192))" in output


def test_simple_cfg_diamond_is_structured_as_if_else():
    entry = block(0x5000, [
        ins(0x5000, "cmp", "ecx, 0"),
        ins(0x5001, "jne", "0x5030", target=0x5030, flow="conditional"),
    ], successors=(0x5030, 0x5010))
    false_branch = block(0x5010, [
        ins(0x5010, "mov", "eax, 1"),
        ins(0x5011, "jmp", "0x5040", target=0x5040, flow="jump"),
    ], successors=(0x5040,))
    true_branch = block(0x5030, [
        ins(0x5030, "mov", "eax, 2"),
        ins(0x5031, "jmp", "0x5040", target=0x5040, flow="jump"),
    ], successors=(0x5040,))
    join = block(0x5040, [
        ins(0x5040, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x5000: "choose"})
    lifter.infer_function_signatures({0x5000: [entry, false_branch, true_branch, join]})
    output = lifter.function_text("choose", 0x5000, [entry, false_branch, true_branch, join])
    assert "if (" in output
    assert "} else {" in output
    assert "goto loc_5030;" not in output
    assert "goto loc_5040;" not in output


def test_simple_natural_loop_is_structured_as_while():
    header = block(0x6000, [
        ins(0x6000, "cmp", "ecx, 0"),
        ins(0x6001, "jne", "0x6020", target=0x6020, flow="conditional"),
    ], successors=(0x6020, 0x6030))
    loop_body = block(0x6020, [
        ins(0x6020, "inc", "eax"),
        ins(0x6021, "jmp", "0x6000", target=0x6000, flow="jump"),
    ], successors=(0x6000,))
    exit_block = block(0x6030, [
        ins(0x6030, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x6000: "count"})
    lifter.infer_function_signatures({0x6000: [header, loop_body, exit_block]})
    output = lifter.function_text("count", 0x6000, [header, loop_body, exit_block])
    assert "while (" in output
    assert "goto loc_6000;" not in output
    assert "loc_6020:" not in output


def test_multiblock_loop_recovers_conditional_break_and_continue():
    header = block(0xA000, [
        ins(0xA000, "cmp", "ecx, 0"),
        ins(0xA001, "jne", "0xA002", target=0xA002, flow="conditional"),
    ], successors=(0xA002, 0xA008))
    early_exit = block(0xA002, [
        ins(0xA002, "cmp", "eax, 0"),
        ins(0xA003, "je", "0xA006", target=0xA006, flow="conditional"),
    ], successors=(0xA006, 0xA004))
    break_check = block(0xA004, [
        ins(0xA004, "cmp", "edx, 0"),
        ins(0xA005, "jne", "0xA008", target=0xA008, flow="conditional"),
    ], successors=(0xA008, 0xA006))
    continue_block = block(0xA006, [
        ins(0xA006, "inc", "eax"),
        ins(0xA007, "jmp", "0xA000", target=0xA000, flow="jump"),
    ], successors=(0xA000,))
    exit_block = block(0xA008, [ins(0xA008, "ret", flow="return")])
    blocks = [header, early_exit, break_check, continue_block, exit_block]
    lifter = PseudocodeLifter(64, function_names={0xA000: "walk"})
    lifter.infer_function_signatures({0xA000: blocks})
    output = lifter.function_text("walk", 0xA000, blocks)
    assert "while (" in output
    assert "if (" in output and "break;" in output
    assert "continue;" in output
    assert "goto loc_A000;" not in output
    assert "loc_A002:" not in output


def test_outer_loop_structures_break_continue_around_inner_cycle():
    outer_header = block(0xA100, [
        ins(0xA100, "cmp", "ecx, 0"),
        ins(0xA101, "jne", "0xA102", target=0xA102, flow="conditional"),
    ], successors=(0xA102, 0xA120))
    inner_header = block(0xA102, [
        ins(0xA102, "cmp", "eax, 0"),
        ins(0xA103, "jne", "0xA110", target=0xA110, flow="conditional"),
    ], successors=(0xA110, 0xA104))
    break_check = block(0xA104, [
        ins(0xA104, "cmp", "edx, 0"),
        ins(0xA105, "jne", "0xA120", target=0xA120, flow="conditional"),
    ], successors=(0xA120, 0xA106))
    continue_block = block(0xA106, [
        ins(0xA106, "jmp", "0xA100", target=0xA100, flow="jump"),
    ], successors=(0xA100,))
    inner_body = block(0xA110, [
        ins(0xA110, "inc", "eax"),
        ins(0xA111, "jmp", "0xA102", target=0xA102, flow="jump"),
    ], successors=(0xA102,))
    exit_block = block(0xA120, [ins(0xA120, "ret", flow="return")])
    blocks = [outer_header, inner_header, break_check, continue_block, inner_body, exit_block]
    lifter = PseudocodeLifter(64, function_names={0xA100: "nested"})
    lifter.infer_function_signatures({0xA100: blocks})
    output = lifter.function_text("nested", 0xA100, blocks)
    assert "while (" in output
    assert "if (" in output and "break;" in output
    assert "continue;" in output
    # The inner cycle remains an explicit CFG loop, preserving its exact edges.
    assert "goto loc_A102;" in output


def test_simple_post_tested_loop_is_structured_as_do_while():
    loop_body = block(0x7000, [
        ins(0x7000, "inc", "eax"),
        ins(0x7001, "cmp", "ecx, 0"),
        ins(0x7002, "jne", "0x7000", target=0x7000, flow="conditional"),
    ], successors=(0x7000, 0x7020))
    exit_block = block(0x7020, [
        ins(0x7020, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x7000: "repeat"})
    lifter.infer_function_signatures({0x7000: [loop_body, exit_block]})
    output = lifter.function_text("repeat", 0x7000, [loop_body, exit_block])
    assert "do {" in output
    assert "} while (" in output
    assert "goto loc_7000;" not in output
    assert "loc_7000:" not in output


def test_simple_dispatch_chain_is_structured_as_switch():
    dispatch_one = block(0x8000, [
        ins(0x8000, "cmp", "ecx, 1"),
        ins(0x8001, "je", "0x8020", target=0x8020, flow="conditional"),
    ], successors=(0x8020, 0x8010))
    dispatch_two = block(0x8010, [
        ins(0x8010, "cmp", "ecx, 2"),
        ins(0x8011, "je", "0x8040", target=0x8040, flow="conditional"),
    ], successors=(0x8040, 0x8030))
    case_one = block(0x8020, [
        ins(0x8020, "mov", "eax, 10"),
        ins(0x8021, "jmp", "0x8050", target=0x8050, flow="jump"),
    ], successors=(0x8050,))
    default = block(0x8030, [
        ins(0x8030, "mov", "eax, 0"),
        ins(0x8031, "jmp", "0x8050", target=0x8050, flow="jump"),
    ], successors=(0x8050,))
    case_two = block(0x8040, [
        ins(0x8040, "mov", "eax, 20"),
        ins(0x8041, "jmp", "0x8050", target=0x8050, flow="jump"),
    ], successors=(0x8050,))
    join = block(0x8050, [ins(0x8050, "ret", flow="return")])
    blocks = [dispatch_one, dispatch_two, case_one, default, case_two, join]
    lifter = PseudocodeLifter(64, function_names={0x8000: "select"})
    lifter.infer_function_signatures({0x8000: blocks})
    output = lifter.function_text("select", 0x8000, blocks)
    assert "switch (ecx)" in output
    assert "case 1:" in output
    assert "case 2:" in output
    assert "default:" in output


def test_switch_case_fallthrough_is_structured_without_break():
    dispatch_one = block(0x8100, [
        ins(0x8100, "cmp", "ecx, 1"),
        ins(0x8101, "je", "0x8120", target=0x8120, flow="conditional"),
    ], successors=(0x8120, 0x8110))
    dispatch_two = block(0x8110, [
        ins(0x8110, "cmp", "ecx, 2"),
        ins(0x8111, "je", "0x8130", target=0x8130, flow="conditional"),
    ], successors=(0x8130, 0x8140))
    case_one = block(0x8120, [
        ins(0x8120, "mov", "eax, 10"),
        ins(0x8121, "jmp", "0x8130", target=0x8130, flow="jump"),
    ], successors=(0x8130,))
    case_two = block(0x8130, [
        ins(0x8130, "add", "eax, 20"),
        ins(0x8131, "jmp", "0x8150", target=0x8150, flow="jump"),
    ], successors=(0x8150,))
    default = block(0x8140, [
        ins(0x8140, "mov", "eax, 0"),
        ins(0x8141, "jmp", "0x8150", target=0x8150, flow="jump"),
    ], successors=(0x8150,))
    join = block(0x8150, [ins(0x8150, "ret", flow="return")])
    blocks = [dispatch_one, dispatch_two, case_one, case_two, default, join]
    lifter = PseudocodeLifter(64, function_names={0x8100: "select_with_fallthrough"})
    lifter.infer_function_signatures({0x8100: blocks})
    output = lifter.function_text("select_with_fallthrough", 0x8100, blocks)
    assert "switch (ecx)" in output
    assert "case 1:" in output
    assert "case 2:" in output
    assert output.count("break;") == 2
    assert "goto loc_8130;" not in output


def test_control_flow_structuring_repeats_for_two_safe_diamonds():
    entry = block(0x9000, [
        ins(0x9000, "cmp", "ecx, 0"),
        ins(0x9001, "jne", "0x9020", target=0x9020, flow="conditional"),
    ], successors=(0x9020, 0x9010))
    first_true = block(0x9010, [
        ins(0x9010, "mov", "eax, 1"),
        ins(0x9011, "jmp", "0x9030", target=0x9030, flow="jump"),
    ], successors=(0x9030,))
    first_false = block(0x9020, [
        ins(0x9020, "mov", "eax, 2"),
        ins(0x9021, "jmp", "0x9030", target=0x9030, flow="jump"),
    ], successors=(0x9030,))
    second_header = block(0x9030, [
        ins(0x9030, "cmp", "edx, 0"),
        ins(0x9031, "jne", "0x9050", target=0x9050, flow="conditional"),
    ], successors=(0x9050, 0x9040))
    second_true = block(0x9040, [
        ins(0x9040, "mov", "ebx, 3"),
        ins(0x9041, "jmp", "0x9060", target=0x9060, flow="jump"),
    ], successors=(0x9060,))
    second_false = block(0x9050, [
        ins(0x9050, "mov", "ebx, 4"),
        ins(0x9051, "jmp", "0x9060", target=0x9060, flow="jump"),
    ], successors=(0x9060,))
    join = block(0x9060, [ins(0x9060, "ret", flow="return")])
    blocks = [entry, first_true, first_false, second_header, second_true, second_false, join]
    lifter = PseudocodeLifter(64, function_names={0x9000: "nested_choose"})
    lifter.infer_function_signatures({0x9000: blocks})
    output = lifter.function_text("nested_choose", 0x9000, blocks)
    assert output.count("if (") >= 2
    assert "} else {" in output



def test_single_branch_if_is_structured_without_goto():
    entry = block(0xB000, [
        ins(0xB000, "cmp", "ecx, 0"),
        ins(0xB001, "je", "0xB020", target=0xB020, flow="conditional"),
    ], successors=(0xB020, 0xB010))
    then_block = block(0xB010, [
        ins(0xB010, "mov", "eax, 42"),
    ], successors=(0xB020,))
    join = block(0xB020, [
        ins(0xB020, "ret", flow="return"),
    ])
    blocks = [entry, then_block, join]
    lifter = PseudocodeLifter(64, function_names={0xB000: "check"})
    lifter.infer_function_signatures({0xB000: blocks})
    output = lifter.function_text("check", 0xB000, blocks)
    assert "if (" in output
    assert "goto loc_B020;" not in output
    assert "loc_B010:" not in output
    assert "loc_B020:" not in output
