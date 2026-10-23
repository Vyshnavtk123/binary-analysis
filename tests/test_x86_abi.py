import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decompiler_engine.models import BasicBlock, Instruction
from decompiler_engine.pseudocode import PseudocodeLifter


def ins(address, mnemonic, operands="", *, target=None, flow="normal"):
    return Instruction(address, 1, "90", mnemonic, operands, ".text", target, None, flow)


def block(start, instructions):
    return BasicBlock(start, instructions[-1].end, instructions, [], ".text")


def test_x86_stdcall_cleanup_and_frame_arguments_are_inferred():
    fn = block(0x1000, [
        ins(0x1000, "push", "ebp"),
        ins(0x1001, "mov", "ebp, esp"),
        ins(0x1002, "mov", "eax, dword ptr [ebp + 8]"),
        ins(0x1003, "add", "eax, dword ptr [ebp + 0xC]"),
        ins(0x1004, "ret", "8", flow="return"),
    ])
    lifter = PseudocodeLifter(32)
    signature = lifter.infer_function_signatures({0x1000: [fn]})[0]
    assert signature["calling_convention"] == "stdcall (heuristic)"
    assert signature["callee_pop_bytes"] == 8
    assert signature["parameter_count"] == 2
    assert signature["stack_parameters"] == ["stack_arg_8", "stack_arg_C"]
    assert "stack_arg_4" not in signature["stack_parameters"]


def test_x86_cdecl_arguments_are_inferred_without_callee_cleanup():
    fn = block(0x1100, [
        ins(0x1100, "push", "ebp"),
        ins(0x1101, "mov", "ebp, esp"),
        ins(0x1102, "mov", "eax, dword ptr [ebp + 8]"),
        ins(0x1103, "cmp", "dword ptr [ebp + 0x10], 0"),
        ins(0x1104, "ret", flow="return"),
    ])
    signature = PseudocodeLifter(32).infer_function_signatures({0x1100: [fn]})[0]
    assert signature["calling_convention"] == "cdecl candidate (heuristic)"
    assert signature["callee_pop_bytes"] == 0
    assert signature["parameter_count"] == 3
    assert signature["stack_parameters"] == ["stack_arg_8", "stack_arg_C", "stack_arg_10"]


def test_x86_lea_frame_setup_normalizes_arguments_to_entry_stack_offsets():
    fn = block(0x1180, [
        ins(0x1180, "push", "ebp"),
        ins(0x1181, "lea", "ebp, [esp + 4]"),
        ins(0x1182, "mov", "eax, dword ptr [ebp + 4]"),
        ins(0x1183, "ret", flow="return"),
    ])
    signature = PseudocodeLifter(32).infer_function_signatures({0x1180: [fn]})[0]
    assert signature["stack_parameters"] == ["stack_arg_8"]
    assert signature["parameter_count"] == 1


def test_x86_lea_frame_with_unknown_displacement_is_not_assumed_as_an_argument_frame():
    fn = block(0x1190, [
        ins(0x1190, "push", "ebp"),
        ins(0x1191, "lea", "ebp, [esp + eax]"),
        ins(0x1192, "mov", "eax, dword ptr [ebp + 8]"),
        ins(0x1193, "ret", flow="return"),
    ])
    signature = PseudocodeLifter(32).infer_function_signatures({0x1190: [fn]})[0]
    assert signature["stack_parameters"] == []
    assert signature["abi_warnings"]


def test_x86_esp_argument_offsets_propagate_through_unambiguous_cfg_edge():
    entry = BasicBlock(0x11A0, 0x11A6, [
        ins(0x11A0, "sub", "esp, 8"),
        ins(0x11A2, "jmp", "0x11A6", target=0x11A6, flow="jump"),
    ], [0x11A6], ".text")
    body = BasicBlock(0x11A6, 0x11A9, [
        ins(0x11A6, "mov", "eax, dword ptr [esp + 12]"),
        ins(0x11A8, "ret", flow="return"),
    ], [], ".text")
    signature = PseudocodeLifter(32).infer_function_signatures({0x11A0: [entry, body]})[0]
    assert signature["stack_parameters"] == ["stack_arg_8"]
    assert signature["abi_evidence"]["esp_argument_offsets_observed"] == ["0x4"]


def test_x86_frame_setup_in_an_alternate_block_is_not_used_as_entry_prologue():
    entry = BasicBlock(0x11B0, 0x11B3, [
        ins(0x11B0, "cmp", "eax, 0"),
        ins(0x11B1, "je", "0x11C0", target=0x11C0, flow="conditional"),
    ], [0x11B3, 0x11C0], ".text")
    fallthrough = BasicBlock(0x11B3, 0x11B4, [ins(0x11B3, "ret", flow="return")], [], ".text")
    alternate = BasicBlock(0x11C0, 0x11C4, [
        ins(0x11C0, "push", "ebp"),
        ins(0x11C1, "mov", "ebp, esp"),
        ins(0x11C2, "mov", "eax, dword ptr [ebp + 8]"),
        ins(0x11C3, "ret", flow="return"),
    ], [], ".text")
    signature = PseudocodeLifter(32).infer_function_signatures(
        {0x11B0: [entry, fallthrough, alternate]}
    )[0]
    assert signature["stack_parameters"] == []


def test_x86_register_argument_survives_branch_where_only_one_path_overwrites_it():
    entry = BasicBlock(0x11D0, 0x11D2, [
        ins(0x11D0, "jne", "0x11E0", target=0x11E0, flow="conditional"),
    ], [0x11D2, 0x11E0], ".text")
    overwrite = BasicBlock(0x11D2, 0x11D4, [
        ins(0x11D2, "mov", "ecx, 0"),
        ins(0x11D3, "jmp", "0x11F0", target=0x11F0, flow="jump"),
    ], [0x11F0], ".text")
    preserve = BasicBlock(0x11E0, 0x11E2, [
        ins(0x11E0, "nop"),
        ins(0x11E1, "jmp", "0x11F0", target=0x11F0, flow="jump"),
    ], [0x11F0], ".text")
    join = BasicBlock(0x11F0, 0x11F2, [
        ins(0x11F0, "mov", "eax, ecx"),
        ins(0x11F1, "ret", flow="return"),
    ], [], ".text")
    signature = PseudocodeLifter(32).infer_function_signatures(
        {0x11D0: [entry, overwrite, preserve, join]}
    )[0]
    assert signature["register_parameters"][0]["register"] == "ecx"


def test_x86_register_is_not_an_argument_if_all_cfg_paths_overwrite_it_first():
    entry = BasicBlock(0x1200, 0x1202, [
        ins(0x1200, "jne", "0x1210", target=0x1210, flow="conditional"),
    ], [0x1202, 0x1210], ".text")
    left = BasicBlock(0x1202, 0x1204, [
        ins(0x1202, "mov", "ecx, 1"),
        ins(0x1203, "jmp", "0x1220", target=0x1220, flow="jump"),
    ], [0x1220], ".text")
    right = BasicBlock(0x1210, 0x1212, [
        ins(0x1210, "mov", "ecx, 2"),
        ins(0x1211, "jmp", "0x1220", target=0x1220, flow="jump"),
    ], [0x1220], ".text")
    join = BasicBlock(0x1220, 0x1222, [
        ins(0x1220, "mov", "eax, ecx"),
        ins(0x1221, "ret", flow="return"),
    ], [], ".text")
    signature = PseudocodeLifter(32).infer_function_signatures(
        {0x1200: [entry, left, right, join]}
    )[0]
    assert signature["register_parameters"] == []


def test_x86_fastcall_register_and_stack_arguments_are_inferred():
    fn = block(0x1200, [
        ins(0x1200, "push", "ebp"),
        ins(0x1201, "mov", "ebp, esp"),
        ins(0x1202, "mov", "dword ptr [ebp - 4], ecx"),
        ins(0x1203, "mov", "dword ptr [ebp - 8], edx"),
        ins(0x1204, "mov", "eax, dword ptr [ebp + 8]"),
        ins(0x1205, "ret", "4", flow="return"),
    ])
    lifter = PseudocodeLifter(32)
    signature = lifter.infer_function_signatures({0x1200: [fn]})[0]
    assert signature["calling_convention"] == "fastcall (heuristic)"
    assert signature["register_parameters"] == [
        {"name": "arg1", "register": "ecx", "type": "uint32_t"},
        {"name": "arg2", "register": "edx", "type": "uint32_t"},
    ]
    assert signature["parameter_count"] == 3
    output = lifter.function_text("f", 0x1200, [fn])
    assert "uint32_t f(uint32_t arg1, uint32_t arg2, uint32_t stack_arg_8)" in output


def test_x86_ecx_receiver_is_reported_as_a_thiscall_candidate():
    fn = block(0x1300, [
        ins(0x1300, "push", "ebp"),
        ins(0x1301, "mov", "ebp, esp"),
        ins(0x1302, "mov", "eax, ecx"),
        ins(0x1303, "add", "eax, dword ptr [ebp + 8]"),
        ins(0x1304, "ret", "4", flow="return"),
    ])
    signature = PseudocodeLifter(32).infer_function_signatures({0x1300: [fn]})[0]
    assert signature["calling_convention"].startswith("thiscall candidate")
    assert signature["register_parameters"] == [{"name": "this", "register": "ecx", "type": "uint32_t"}]
    assert signature["parameter_count"] == 2


def test_x86_thiscall_candidate_forwards_ecx_receiver_and_stack_argument():
    callee = block(0x1350, [
        ins(0x1350, "push", "ebp"),
        ins(0x1351, "mov", "ebp, esp"),
        ins(0x1352, "mov", "eax, dword ptr [ecx + 4]"),
        ins(0x1353, "add", "eax, dword ptr [ebp + 8]"),
        ins(0x1354, "ret", "4", flow="return"),
    ])
    caller = block(0x2350, [
        ins(0x2350, "mov", "ecx, 0x5000"),
        ins(0x2351, "push", "7"),
        ins(0x2352, "call", "0x1350", target=0x1350, flow="call"),
        ins(0x2353, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(32, function_names={0x1350: "method", 0x2350: "caller"})
    signatures = lifter.infer_function_signatures({0x1350: [callee], 0x2350: [caller]})
    receiver_signature = next(row for row in signatures if row["address"] == "0x1350")
    assert receiver_signature["calling_convention"].startswith("thiscall candidate")
    assert receiver_signature["register_parameters"] == [
        {"name": "this", "register": "ecx", "type": "uintptr_t"}
    ]
    output = lifter.function_text("caller", 0x2350, [caller])
    assert "method((uintptr_t)(20480), 7)" in output


def test_x86_direct_call_reconstructs_right_to_left_push_arguments():
    callee = block(0x2000, [
        ins(0x2000, "push", "ebp"),
        ins(0x2001, "mov", "ebp, esp"),
        ins(0x2002, "mov", "eax, dword ptr [ebp + 8]"),
        ins(0x2003, "add", "eax, dword ptr [ebp + 0xC]"),
        ins(0x2004, "ret", flow="return"),
    ])
    caller = block(0x3000, [
        ins(0x3000, "push", "2"),
        ins(0x3001, "push", "1"),
        ins(0x3002, "call", "0x2000", target=0x2000, flow="call"),
        ins(0x3003, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(32, function_names={0x2000: "sum2", 0x3000: "caller"})
    lifter.infer_function_signatures({0x2000: [callee], 0x3000: [caller]})
    output = lifter.function_text("caller", 0x3000, [caller])
    assert "sum2(1, 2)" in output


def test_x86_direct_call_reconstructs_push_arguments_across_blocks():
    callee = block(0x2050, [
        ins(0x2050, "push", "ebp"),
        ins(0x2051, "mov", "ebp, esp"),
        ins(0x2052, "mov", "eax, dword ptr [ebp + 8]"),
        ins(0x2053, "add", "eax, dword ptr [ebp + 0xC]"),
        ins(0x2054, "ret", flow="return"),
    ])
    setup = BasicBlock(0x3050, 0x3052, [
        ins(0x3050, "push", "2"),
        ins(0x3051, "jmp", "0x3060", target=0x3060, flow="jump"),
    ], [0x3060], ".text")
    second_push = BasicBlock(0x3060, 0x3062, [
        ins(0x3060, "push", "1"),
        ins(0x3061, "jmp", "0x3070", target=0x3070, flow="jump"),
    ], [0x3070], ".text")
    invoke = block(0x3070, [
        ins(0x3070, "call", "0x2050", target=0x2050, flow="call"),
        ins(0x3071, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(32, function_names={0x2050: "sum2", 0x3050: "caller"})
    lifter.infer_function_signatures({0x2050: [callee], 0x3050: [setup, second_push, invoke]})
    output = lifter.function_text("caller", 0x3050, [setup, second_push, invoke])
    assert "sum2(1, 2)" in output


def test_x86_direct_call_reconstructs_esp_store_arguments_across_blocks():
    callee = block(0x2060, [
        ins(0x2060, "push", "ebp"),
        ins(0x2061, "mov", "ebp, esp"),
        ins(0x2062, "mov", "eax, dword ptr [ebp + 8]"),
        ins(0x2063, "add", "eax, dword ptr [ebp + 0xC]"),
        ins(0x2064, "ret", "8", flow="return"),
    ])
    reserve = BasicBlock(0x3060, 0x3063, [
        ins(0x3060, "sub", "esp, 8"),
        ins(0x3061, "mov", "dword ptr [esp + 4], 2"),
        ins(0x3062, "jmp", "0x3070", target=0x3070, flow="jump"),
    ], [0x3070], ".text")
    first_argument = BasicBlock(0x3070, 0x3072, [
        ins(0x3070, "mov", "dword ptr [esp], 1"),
        ins(0x3071, "jmp", "0x3080", target=0x3080, flow="jump"),
    ], [0x3080], ".text")
    invoke = block(0x3080, [
        ins(0x3080, "call", "0x2060", target=0x2060, flow="call"),
        ins(0x3081, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(32, function_names={0x2060: "sum2", 0x3060: "caller"})
    lifter.infer_function_signatures({0x2060: [callee], 0x3060: [reserve, first_argument, invoke]})
    output = lifter.function_text("caller", 0x3060, [reserve, first_argument, invoke])
    assert "sum2(1, 2)" in output


def test_x86_push_argument_joins_keep_agreement_and_widen_conflicts():
    callee = block(0x2080, [
        ins(0x2080, "push", "ebp"),
        ins(0x2081, "mov", "ebp, esp"),
        ins(0x2082, "mov", "eax, dword ptr [ebp + 8]"),
        ins(0x2083, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(32, function_names={0x2080: "take1"})
    lifter.infer_function_signatures({0x2080: [callee]})

    entry = BasicBlock(0x3180, 0x3182, [
        ins(0x3180, "push", "2"),
        ins(0x3181, "jne", "0x3190", target=0x3190, flow="conditional"),
    ], [0x3190, 0x3182], ".text")
    left = BasicBlock(0x3182, 0x3183, [ins(0x3182, "jmp", "0x31A0", target=0x31A0, flow="jump")], [0x31A0], ".text")
    right = BasicBlock(0x3190, 0x3191, [ins(0x3190, "jmp", "0x31A0", target=0x31A0, flow="jump")], [0x31A0], ".text")
    invoke = block(0x31A0, [ins(0x31A0, "call", "0x2080", target=0x2080, flow="call")])
    agreed = lifter.function_text("agreed", 0x3180, [entry, left, right, invoke])
    assert "take1(2)" in agreed

    conflict_entry = BasicBlock(0x31B0, 0x31B1, [
        ins(0x31B0, "jne", "0x31C0", target=0x31C0, flow="conditional"),
    ], [0x31C0, 0x31B1], ".text")
    conflict_left = BasicBlock(0x31B1, 0x31B3, [
        ins(0x31B1, "push", "1"),
        ins(0x31B2, "jmp", "0x31D0", target=0x31D0, flow="jump"),
    ], [0x31D0], ".text")
    conflict_right = BasicBlock(0x31C0, 0x31C2, [
        ins(0x31C0, "push", "3"),
        ins(0x31C1, "jmp", "0x31D0", target=0x31D0, flow="jump"),
    ], [0x31D0], ".text")
    conflict_invoke = block(0x31D0, [ins(0x31D0, "call", "0x2080", target=0x2080, flow="call")])
    conflicted = lifter.function_text(
        "conflicted", 0x31B0, [conflict_entry, conflict_left, conflict_right, conflict_invoke]
    )
    assert "take1(unknown_stack_arg_1)" in conflicted


def test_x86_push_joins_preserve_agreed_older_arguments_and_number_unknowns_by_parameter():
    callee = block(0x20A0, [
        ins(0x20A0, "mov", "eax, dword ptr [esp + 4]"),
        ins(0x20A1, "mov", "edx, dword ptr [esp + 8]"),
        ins(0x20A2, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(32, function_names={0x20A0: "take_two"})
    lifter.infer_function_signatures({0x20A0: [callee]})

    entry = BasicBlock(0x32A0, 0x32A1, [
        ins(0x32A0, "jne", "0x32B0", target=0x32B0, flow="conditional"),
    ], [0x32B0, 0x32A1], ".text")
    left = BasicBlock(0x32A1, 0x32A4, [
        ins(0x32A1, "push", "2"),
        ins(0x32A2, "push", "1"),
        ins(0x32A3, "jmp", "0x32C0", target=0x32C0, flow="jump"),
    ], [0x32C0], ".text")
    right = BasicBlock(0x32B0, 0x32B2, [
        ins(0x32B0, "push", "2"),
        ins(0x32B1, "jmp", "0x32C0", target=0x32C0, flow="jump"),
    ], [0x32C0], ".text")
    invoke = block(0x32C0, [
        ins(0x32C0, "call", "0x20A0", target=0x20A0, flow="call"),
    ])
    output = lifter.function_text("caller", 0x32A0, [entry, left, right, invoke])
    assert "take_two(unknown_stack_arg_1, 2)" in output


def test_x86_direct_call_marks_missing_known_stack_arguments():
    callee = block(0x2100, [
        ins(0x2100, "mov", "eax, dword ptr [esp + 4]"),
        ins(0x2101, "mov", "edx, dword ptr [esp + 8]"),
        ins(0x2102, "ret", flow="return"),
    ])
    caller = block(0x3100, [
        ins(0x3100, "push", "2"),
        ins(0x3101, "call", "0x2100", target=0x2100, flow="call"),
        ins(0x3102, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(32, function_names={0x2100: "sum2"})
    lifter.infer_function_signatures({0x2100: [callee], 0x3100: [caller]})
    output = lifter.function_text("caller", 0x3100, [caller])
    assert "sum2(unknown_stack_arg_1, 2)" in output


def test_x86_caller_cleanup_discards_stale_arguments_before_next_call():
    callee = block(0x2200, [
        ins(0x2200, "mov", "eax, dword ptr [esp + 4]"),
        ins(0x2201, "ret", flow="return"),
    ])
    caller = block(0x3200, [
        ins(0x3200, "push", "22"),
        ins(0x3201, "push", "11"),
        ins(0x3202, "call", "0x4000", target=0x4000, flow="call"),
        ins(0x3203, "add", "esp, 8"),
        ins(0x3204, "call", "0x2200", target=0x2200, flow="call"),
        ins(0x3205, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(32, function_names={0x2200: "takes_one", 0x4000: "unknown_target"})
    lifter.infer_function_signatures({0x2200: [callee], 0x3200: [caller]})
    output = lifter.function_text("caller", 0x3200, [caller])
    assert "takes_one(unknown_stack_arg_1)" in output
    assert "takes_one(22)" not in output


def test_x86_esp_arguments_normalize_across_local_stack_allocation():
    fn = block(0x4000, [
        ins(0x4000, "sub", "esp, 0x10"),
        ins(0x4001, "mov", "eax, dword ptr [esp + 0x14]"),
        ins(0x4002, "ret", flow="return"),
    ])
    signature = PseudocodeLifter(32).infer_function_signatures({0x4000: [fn]})[0]
    assert signature["stack_parameters"] == ["stack_arg_8"]
    assert signature["abi_evidence"]["frame_argument_offsets_observed"] == ["0x8"]
    assert signature["abi_evidence"]["esp_argument_offsets_observed"] == ["0x4"]
    assert signature["abi_warnings"] == []


def test_x86_esp_arguments_account_for_saved_register_pushes():
    fn = block(0x4100, [
        ins(0x4100, "push", "ebx"),
        ins(0x4101, "push", "esi"),
        ins(0x4102, "mov", "eax, dword ptr [esp + 0xC]"),
        ins(0x4103, "ret", flow="return"),
    ])
    signature = PseudocodeLifter(32).infer_function_signatures({0x4100: [fn]})[0]
    assert signature["stack_parameters"] == ["stack_arg_8"]
    assert signature["abi_evidence"]["esp_argument_offsets_observed"] == ["0x4"]


def test_x86_esp_argument_offsets_follow_pops_that_advance_the_stack():
    fn = block(0x4150, [
        ins(0x4150, "pop", "eax"),
        ins(0x4151, "mov", "edx, dword ptr [esp]"),
        ins(0x4152, "ret", flow="return"),
    ])
    signature = PseudocodeLifter(32).infer_function_signatures({0x4150: [fn]})[0]
    assert signature["stack_parameters"] == ["stack_arg_8"]
    assert signature["abi_evidence"]["esp_argument_offsets_observed"] == ["0x4"]


def test_x86_esp_argument_analysis_stops_at_unknown_alignment_and_branches():
    aligned = block(0x4200, [
        ins(0x4200, "and", "esp, -16"),
        ins(0x4201, "mov", "eax, dword ptr [esp + 4]"),
        ins(0x4202, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(32)
    aligned_signature = lifter.infer_function_signatures({0x4200: [aligned]})[0]
    assert aligned_signature["stack_parameters"] == []
    assert any("not normalized" in warning for warning in aligned_signature["abi_warnings"])

    branched = block(0x4300, [
        ins(0x4300, "cmp", "ecx, 0"),
        ins(0x4301, "je", "0x4310", target=0x4310, flow="conditional"),
        ins(0x4302, "mov", "eax, dword ptr [esp + 4]"),
        ins(0x4303, "ret", flow="return"),
    ])
    branch_signature = lifter.infer_function_signatures({0x4300: [branched]})[0]
    assert branch_signature["stack_parameters"] == []
