import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decompiler_engine.models import BasicBlock, Instruction
from decompiler_engine.pseudocode import PseudocodeLifter


def ins(address, mnemonic, operands="", *, target=None, flow="normal"):
    return Instruction(address, 1, "90", mnemonic, operands, ".text", target, flow=flow)


def block(start, instructions, successors=()):
    return BasicBlock(start, instructions[-1].end, instructions, list(successors), ".text")


def test_x64_rbp_argument_offsets_normalize_saved_rbp_and_shadow_space():
    fn = block(0x5000, [
        ins(0x5000, "push", "rbp"),
        ins(0x5001, "mov", "rbp, rsp"),
        ins(0x5002, "mov", "rax, qword ptr [rbp + 0x30]"),
        ins(0x5003, "ret", flow="return"),
    ])
    signature = PseudocodeLifter(64).infer_function_signatures({0x5000: [fn]})[0]
    assert signature["stack_parameters"] == ["stack_arg_28"]
    assert signature["shadow_space_bytes"] == 32
    lifter = PseudocodeLifter(64)
    lifter.infer_function_signatures({0x5000: [fn]})
    output = lifter.function_text("f", 0x5000, [fn])
    assert "uint64_t stack_arg_28" in output
    assert "v_rax = stack_arg_28;" in output


def test_x64_constant_lea_frame_setup_normalizes_stack_arguments():
    fn = block(0x5080, [
        ins(0x5080, "push", "rbp"),
        ins(0x5081, "lea", "rbp, [rsp + 8]"),
        ins(0x5082, "mov", "rax, qword ptr [rbp + 0x28]"),
        ins(0x5083, "ret", flow="return"),
    ])
    signature = PseudocodeLifter(64).infer_function_signatures({0x5080: [fn]})[0]
    assert signature["stack_parameters"] == ["stack_arg_28"]


def test_x64_rsp_argument_offsets_normalize_local_allocation():
    fn = block(0x5100, [
        ins(0x5100, "sub", "rsp, 0x40"),
        ins(0x5101, "mov", "rax, qword ptr [rsp + 0x68]"),
        ins(0x5102, "ret", flow="return"),
    ])
    signature = PseudocodeLifter(64).infer_function_signatures({0x5100: [fn]})[0]
    assert signature["stack_parameters"] == ["stack_arg_28"]
    lifter = PseudocodeLifter(64)
    lifter.infer_function_signatures({0x5100: [fn]})
    output = lifter.function_text("f", 0x5100, [fn])
    assert "v_rax = stack_arg_28;" in output


def test_x64_rsp_argument_offset_propagates_across_cfg_edge_with_known_depth():
    entry = block(0x5180, [
        ins(0x5180, "sub", "rsp, 0x40"),
        ins(0x5181, "jmp", "0x5182", target=0x5182, flow="jump"),
    ], successors=[0x5182])
    body = block(0x5182, [
        ins(0x5182, "mov", "rax, qword ptr [rsp + 0x68]"),
        ins(0x5183, "ret", flow="return"),
    ])
    signature = PseudocodeLifter(64).infer_function_signatures({0x5180: [entry, body]})[0]
    assert signature["stack_parameters"] == ["stack_arg_28"]


def test_x64_rsp_argument_offset_is_unknown_when_join_depths_disagree():
    entry = block(0x5280, [
        ins(0x5280, "jne", "0x5290", target=0x5290, flow="conditional"),
    ], successors=[0x5281, 0x5290])
    left = block(0x5281, [
        ins(0x5281, "sub", "rsp, 0x20"),
        ins(0x5282, "jmp", "0x52A0", target=0x52A0, flow="jump"),
    ], successors=[0x52A0])
    right = block(0x5290, [
        ins(0x5290, "jmp", "0x52A0", target=0x52A0, flow="jump"),
    ], successors=[0x52A0])
    join = block(0x52A0, [
        ins(0x52A0, "mov", "rax, qword ptr [rsp + 0x48]"),
        ins(0x52A1, "ret", flow="return"),
    ])
    signature = PseudocodeLifter(64).infer_function_signatures(
        {0x5280: [entry, left, right, join]}
    )[0]
    assert signature["stack_parameters"] == []


def test_x64_shadow_space_and_local_rsp_slots_are_not_parameters():
    fn = block(0x5200, [
        ins(0x5200, "sub", "rsp, 0x40"),
        ins(0x5201, "mov", "rax, qword ptr [rsp + 0x28]"),
        ins(0x5202, "mov", "rcx, qword ptr [rsp + 0x20]"),
        ins(0x5203, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64)
    signature = lifter.infer_function_signatures({0x5200: [fn]})[0]
    assert signature["stack_parameters"] == []
    output = lifter.function_text("f", 0x5200, [fn])
    assert "v_rax = local_18;" in output
    assert "v_rcx = local_20;" in output


def test_x64_pointer_register_argument_is_typed_and_cast_at_direct_call():
    callee = block(0x5350, [
        ins(0x5350, "mov", "eax, dword ptr [rcx + 8]"),
        ins(0x5351, "ret", flow="return"),
    ])
    caller = block(0x6350, [
        ins(0x6350, "mov", "rcx, 0x1000"),
        Instruction(0x6351, 1, "90", "call", "0x5350", ".text", 0x5350, flow="call"),
        ins(0x6352, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x5350: "read_object"})
    lifter.infer_function_signatures({0x5350: [callee], 0x6350: [caller]})
    signature = lifter.function_signatures[0x5350]
    assert signature["register_parameters"] == [
        {"name": "arg1", "register": "rcx", "position": 0, "type": "uintptr_t"}
    ]
    output = lifter.function_text("caller", 0x6350, [caller])
    assert "read_object((uintptr_t)(4096))" in output


def test_x64_rsp_stack_recovery_stops_after_alignment_mask():
    fn = block(0x5300, [
        ins(0x5300, "and", "rsp, -16"),
        ins(0x5301, "mov", "rax, qword ptr [rsp + 0x28]"),
        ins(0x5302, "ret", flow="return"),
    ])
    signature = PseudocodeLifter(64).infer_function_signatures({0x5300: [fn]})[0]
    assert signature["stack_parameters"] == []


def test_x64_direct_call_pairs_outgoing_stack_slot_with_known_callee_parameter():
    callee = block(0x6000, [
        ins(0x6000, "push", "rbp"),
        ins(0x6001, "mov", "rbp, rsp"),
        ins(0x6002, "mov", "rax, rcx"),
        ins(0x6003, "mov", "rax, rdx"),
        ins(0x6004, "mov", "rax, r8"),
        ins(0x6005, "mov", "rax, r9"),
        ins(0x6006, "mov", "rax, qword ptr [rbp + 0x30]"),
        ins(0x6007, "ret", flow="return"),
    ])
    caller = block(0x6100, [
        ins(0x6100, "sub", "rsp, 0x28"),
        ins(0x6101, "mov", "qword ptr [rsp + 0x20], 5"),
        Instruction(0x6102, 1, "90", "call", "0x6000", ".text", 0x6000, flow="call"),
        ins(0x6103, "add", "rsp, 0x28"),
        ins(0x6104, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x6000: "callee"})
    lifter.infer_function_signatures({0x6000: [callee], 0x6100: [caller]})
    assert lifter.function_signatures[0x6000]["stack_parameters"] == ["stack_arg_28"]
    output = lifter.function_text("caller", 0x6100, [caller])
    assert "callee(v_rcx, v_rdx, v_r8, v_r9, 5)" in output


def test_x64_direct_call_pairs_push_ordered_stack_arguments_after_shadow_space():
    callee = block(0x6200, [
        ins(0x6200, "push", "rbp"),
        ins(0x6201, "mov", "rbp, rsp"),
        ins(0x6202, "mov", "rax, rcx"),
        ins(0x6203, "mov", "rax, rdx"),
        ins(0x6204, "mov", "rax, r8"),
        ins(0x6205, "mov", "rax, r9"),
        ins(0x6206, "mov", "rax, qword ptr [rbp + 0x30]"),
        ins(0x6207, "mov", "rax, qword ptr [rbp + 0x38]"),
        ins(0x6208, "ret", flow="return"),
    ])
    caller = block(0x6300, [
        ins(0x6300, "push", "6"),
        ins(0x6301, "push", "5"),
        ins(0x6302, "sub", "rsp, 0x20"),
        Instruction(0x6303, 1, "90", "call", "0x6200", ".text", 0x6200, flow="call"),
        ins(0x6304, "add", "rsp, 0x30"),
        ins(0x6305, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x6200: "callee"})
    lifter.infer_function_signatures({0x6200: [callee], 0x6300: [caller]})
    assert lifter.function_signatures[0x6200]["stack_parameters"] == ["stack_arg_28", "stack_arg_30"]
    output = lifter.function_text("caller", 0x6300, [caller])
    assert "callee(v_rcx, v_rdx, v_r8, v_r9, 5, 6)" in output


def test_x64_direct_call_recovers_constant_outgoing_slot_across_cfg_edge():
    callee = block(0x6400, [
        ins(0x6400, "push", "rbp"),
        ins(0x6401, "mov", "rbp, rsp"),
        ins(0x6402, "mov", "rax, rcx"),
        ins(0x6403, "mov", "rax, rdx"),
        ins(0x6404, "mov", "rax, r8"),
        ins(0x6405, "mov", "rax, r9"),
        ins(0x6406, "mov", "rax, qword ptr [rbp + 0x30]"),
        ins(0x6407, "ret", flow="return"),
    ])
    prepare = block(0x6500, [
        ins(0x6500, "mov", "rax, 5"),
        ins(0x6501, "mov", "rbx, rax"),
        ins(0x6502, "add", "rbx, 2"),
        ins(0x6503, "jmp", "0x6504", target=0x6504, flow="jump"),
    ], successors=(0x6504,))
    setup = block(0x6504, [
        ins(0x6504, "sub", "rsp, 0x28"),
        ins(0x6505, "mov", "qword ptr [rsp + 0x20], rbx"),
        ins(0x6506, "jmp", "0x6510", target=0x6510, flow="jump"),
    ], successors=(0x6510,))
    invoke = block(0x6510, [
        Instruction(0x6510, 1, "90", "call", "0x6400", ".text", 0x6400, flow="call"),
        ins(0x6511, "add", "rsp, 0x28"),
        ins(0x6512, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x6400: "callee"})
    lifter.infer_function_signatures({0x6400: [callee], 0x6500: [prepare, setup, invoke]})
    assert lifter.function_signatures[0x6400]["stack_parameters"] == ["stack_arg_28"]
    output = lifter.function_text("caller", 0x6500, [prepare, setup, invoke])
    assert "callee(v_rcx, v_rdx, v_r8, v_r9, 7)" in output


def test_x64_outgoing_slot_tracks_copied_argument_register_across_blocks():
    callee = block(0x6800, [
        ins(0x6800, "push", "rbp"),
        ins(0x6801, "mov", "rbp, rsp"),
        ins(0x6802, "mov", "rax, rcx"),
        ins(0x6803, "mov", "rax, rdx"),
        ins(0x6804, "mov", "rax, r8"),
        ins(0x6805, "mov", "rax, r9"),
        ins(0x6806, "mov", "rax, qword ptr [rbp + 0x30]"),
        ins(0x6807, "ret", flow="return"),
    ])
    prepare = block(0x6900, [
        ins(0x6900, "mov", "rax, rcx"),
        ins(0x6901, "jmp", "0x6902", target=0x6902, flow="jump"),
    ], successors=(0x6902,))
    setup = block(0x6902, [
        ins(0x6902, "sub", "rsp, 0x28"),
        ins(0x6903, "mov", "qword ptr [rsp + 0x20], rax"),
        ins(0x6904, "jmp", "0x6910", target=0x6910, flow="jump"),
    ], successors=(0x6910,))
    invoke = block(0x6910, [
        Instruction(0x6910, 1, "90", "call", "0x6800", ".text", 0x6800, flow="call"),
        ins(0x6911, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x6800: "callee"})
    lifter.infer_function_signatures({0x6800: [callee], 0x6900: [prepare, setup, invoke]})
    output = lifter.function_text("caller", 0x6900, [prepare, setup, invoke])
    assert "callee(arg1, v_rdx, v_r8, v_r9, arg1)" in output


def test_x64_outgoing_slot_is_unknown_after_conflicting_cfg_paths():
    callee = block(0x6600, [
        ins(0x6600, "push", "rbp"),
        ins(0x6601, "mov", "rbp, rsp"),
        ins(0x6602, "mov", "rax, rcx"),
        ins(0x6603, "mov", "rax, rdx"),
        ins(0x6604, "mov", "rax, r8"),
        ins(0x6605, "mov", "rax, r9"),
        ins(0x6606, "mov", "rax, qword ptr [rbp + 0x30]"),
        ins(0x6607, "ret", flow="return"),
    ])
    dispatch = block(0x6700, [
        ins(0x6700, "sub", "rsp, 0x28"),
        ins(0x6701, "cmp", "ecx, 0"),
        ins(0x6702, "jne", "0x6705", target=0x6705, flow="conditional"),
    ], successors=(0x6705, 0x6703))
    left = block(0x6703, [
        ins(0x6703, "mov", "qword ptr [rsp + 0x20], 5"),
        ins(0x6704, "jmp", "0x6707", target=0x6707, flow="jump"),
    ], successors=(0x6707,))
    right = block(0x6705, [
        ins(0x6705, "mov", "qword ptr [rsp + 0x20], 6"),
        ins(0x6706, "jmp", "0x6707", target=0x6707, flow="jump"),
    ], successors=(0x6707,))
    invoke = block(0x6707, [
        Instruction(0x6707, 1, "90", "call", "0x6600", ".text", 0x6600, flow="call"),
        ins(0x6708, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x6600: "callee"})
    lifter.infer_function_signatures({0x6600: [callee], 0x6700: [dispatch, left, right, invoke]})
    output = lifter.function_text("caller", 0x6700, [dispatch, left, right, invoke])
    assert "callee(arg1, v_rdx, v_r8, v_r9, unknown_stack_arg_5)" in output
    assert "callee(arg1, v_rdx, v_r8, v_r9, 5)" not in output
    assert "callee(arg1, v_rdx, v_r8, v_r9, 6)" not in output
    right.instructions[0].operands = "qword ptr [rsp + 0x20], 5"
    agreeing_output = lifter.function_text("caller", 0x6700, [dispatch, left, right, invoke])
    assert "callee(arg1, v_rdx, v_r8, v_r9, 5)" in agreeing_output


def test_x64_scalar_float_register_arguments_recover_and_flow_to_calls():
    callee = block(0x7000, [
        ins(0x7000, "movss", "xmm2, xmm0"),
        ins(0x7001, "ret", flow="return"),
    ])
    caller = block(0x7100, [
        ins(0x7100, "movss", "xmm0, xmm1"),
        Instruction(0x7101, 1, "90", "call", "0x7000", ".text", 0x7000, flow="call"),
        ins(0x7102, "ret", flow="return"),
    ])
    lifter = PseudocodeLifter(64, function_names={0x7000: "consume_float", 0x7100: "caller"})
    signatures = lifter.infer_function_signatures({0x7000: [callee], 0x7100: [caller]})
    callee_signature = signatures[0]
    assert callee_signature["register_parameters"] == [
        {"name": "arg1", "register": "xmm0", "position": 0, "type": "float"}
    ]
    assert callee_signature["parameter_count"] == 1
    output = lifter.function_text("caller", 0x7100, [caller])
    assert "consume_float(arg2)" in output
