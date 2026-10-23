import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decompiler_engine.models import BasicBlock, Instruction
from decompiler_engine.ir import lift_blocks, lift_instruction, normalize_operand
from decompiler_engine.ssa import construct_ssa


def ins(address, mnemonic, operands="", flow="normal", target=None):
    return Instruction(address, 1, "90", mnemonic, operands, ".text", target, None, flow, None)


def test_ir_normalizes_register_aliases_and_constants():
    row = lift_instruction(ins(0x1000, "mov", "eax, 5"), bits=64, block_start=0x1000)
    assert row["opcode"] == "copy"
    assert row["destination"] == "RAX"  # EAX is a subregister of RAX on x64
    assert row["inputs"] == ["const:5"]
    assert row["width"] == 32
    assert row["destination_type"] == "uint32_t"
    assert row["input_types"] == ["int"]


def test_semantic_schema_exposes_register_memory_flag_and_control_effects():
    copy = lift_instruction(ins(0x1000, "mov", "eax, ebx"), bits=64)
    assert copy["machine_bytes"] == "90"
    assert copy["semantics"]["register_reads"] == ["RBX"]
    assert copy["semantics"]["register_writes"] == ["RAX"]
    assert copy["semantics"]["result_width_bits"] == 32

    load = lift_instruction(ins(0x1001, "mov", "eax, dword ptr [rbx + 8]"), bits=64)
    assert load["semantics"]["memory_effects"]["reads"]
    assert load["semantics"]["register_reads"] == ["RBX"]
    assert load["semantics"]["register_writes"] == ["RAX"]

    store = lift_instruction(ins(0x1006, "mov", "dword ptr [rbx + 8], eax"), bits=64)
    assert store["semantics"]["memory_effects"]["reads"] == []
    assert store["semantics"]["memory_effects"]["writes"]
    assert store["semantics"]["register_reads"] == ["RAX", "RBX"]

    arithmetic = lift_instruction(ins(0x1002, "add", "eax, 5"), bits=64)
    assert arithmetic["semantics"]["flag_effects"]["writes"]

    branch = lift_instruction(ins(0x1003, "jne", "0x2000", flow="conditional", target=0x2000), bits=64)
    assert branch["semantics"]["control_flow_effects"]["condition"] == {"ZF": False}
    assert "FLAGS" in branch["semantics"]["register_reads"]

    call = lift_instruction(ins(0x1004, "call", "rax", flow="call"), bits=64)
    assert call["semantics"]["control_flow_effects"]["indirect"] is True
    assert call["semantics"]["call_effects"]["may_write_memory"] is True


def test_unknown_instruction_has_explicit_opaque_semantics():
    row = lift_instruction(ins(0x1005, "madeup", "rax, [rbx]"), bits=64)
    semantics = row["semantics"]
    assert semantics["modeled"] is False
    assert semantics["opcode"] == "opaque"
    assert semantics["unsupported_reason"]
    assert semantics["memory_effects"]["unknown"] is True
    assert semantics["flag_effects"]["unknown"] is True


def test_byte_shuffle_vector_instruction_has_typed_ir_effects():
    row = lift_instruction(ins(0x1002, "pshufb", "xmm0, xmm1"), bits=64)
    assert row["opcode"] == "vector_shuffle"
    assert row["destination"] == "XMM0"
    assert row["defs"] == ["XMM0"]
    assert row["width"] == 128
    assert row["shuffle_control"] == "byte_control"
    assert row["control_high_bit_zeroes_output"] is True


def test_common_integer_simd_operations_have_typed_vector_ir():
    row = lift_instruction(ins(0x1003, "vpaddd", "ymm0, ymm1, ymm2"), bits=64)
    assert row["opcode"] == "vector_binary"
    assert row["destination"] == "YMM0"
    assert row["defs"] == ["YMM0"]
    assert row["inputs"] == ["YMM1", "YMM2"]
    assert row["width"] == 256
    assert row["lane_width"] == 32
    assert row["destination_type"] == "vector256_t"
    shift = lift_instruction(ins(0x1004, "psrad", "xmm0, 1"), bits=64)
    assert shift["opcode"] == "vector_shift"
    assert shift["lane_width"] == 32
    byte_shift = lift_instruction(ins(0x1005, "vpslldq", "ymm0, ymm1, 3"), bits=64)
    assert byte_shift["opcode"] == "vector_shift"
    assert byte_shift["destination"] == "YMM0"
    assert byte_shift["lane_width"] == 128


def test_count_branches_and_loop_ir_use_architectural_counter_widths():
    jcxz = lift_instruction(ins(0x1006, "jcxz", "0x1100", "conditional", 0x1100), bits=64)
    assert jcxz["opcode"] == "conditional_branch"
    assert jcxz["inputs"] == ["CX"]
    jecxz = lift_instruction(ins(0x1007, "jecxz", "0x1100", "conditional", 0x1100), bits=64)
    assert jecxz["inputs"] == ["ECX"]
    loop = lift_instruction(ins(0x1008, "loop", "0x1100", "conditional", 0x1100), bits=32)
    assert loop["opcode"] == "loop_branch"
    assert loop["inputs"] == ["ECX"]
    assert loop["defs"] == ["ECX"]
    assert loop["width"] == 32
    loopz = lift_instruction(ins(0x1009, "loopz", "0x1100", "conditional", 0x1100), bits=64)
    assert loopz["inputs"] == ["RCX", "FLAGS"]


def test_common_x87_stack_instructions_have_explicit_ir_effects():
    load = lift_instruction(ins(0x1010, "fld1"), bits=64)
    assert load["opcode"] == "x87_push"
    assert load["destination"] == "ST0"
    assert load["defs"] == ["X87_STACK", "ST0"]
    assert load["stack_effect"] == 1
    add = lift_instruction(ins(0x1011, "faddp", "st(1), st(0)"), bits=64)
    assert add["opcode"] == "x87_binary"
    assert add["destination"] == "ST1"
    assert add["inputs"] == ["ST1", "ST0"]
    assert add["stack_effect"] == -1
    store = lift_instruction(ins(0x1012, "fstp", "qword ptr [rbp - 8]"), bits=64)
    assert store["opcode"] == "x87_store"
    assert store["stack_effect"] == -1
    assert "X87_STACK" in store["defs"]
    integer_load = lift_instruction(ins(0x1013, "fild", "dword ptr [rbp - 4]"), bits=64)
    assert integer_load["opcode"] == "x87_integer_load"
    assert integer_load["stack_effect"] == 1
    integer_store = lift_instruction(ins(0x1014, "fisttp", "dword ptr [rbp - 8]"), bits=64)
    assert integer_store["opcode"] == "x87_integer_store"
    assert integer_store["rounding_mode"] == "truncate"
    fcmov = lift_instruction(ins(0x1015, "fcmovb", "st(0), st(2)"), bits=64)
    assert fcmov["opcode"] == "x87_conditional_copy"
    assert fcmov["inputs"] == ["ST0", "ST2", "FLAGS"]


def test_ir_detects_stack_locals_and_binary_operations():
    store = lift_instruction(ins(0x1000, "mov", "dword ptr [rbp - 4], eax"), bits=64, block_start=0x1000)
    calc = lift_instruction(ins(0x1001, "add", "eax, 3"), bits=64, block_start=0x1000)
    assert store["destination"] == "STACK_RBP_M4"
    assert store["inputs"] == ["RAX"]
    assert calc["opcode"] == "binary"
    assert calc["expression"] == "(RAX + const:3)"


def test_ssa_places_phi_at_diamond_join_and_versions_values():
    blocks = [
        BasicBlock(0x100, 0x102, [ins(0x100, "mov", "eax, 1"), ins(0x101, "jne", "0x120", "conditional", 0x120)], [0x110, 0x120], ".text"),
        BasicBlock(0x110, 0x112, [ins(0x110, "mov", "eax, 2"), ins(0x111, "jmp", "0x130", "jump", 0x130)], [0x130], ".text"),
        BasicBlock(0x120, 0x122, [ins(0x120, "mov", "eax, 3"), ins(0x121, "jmp", "0x130", "jump", 0x130)], [0x130], ".text"),
        BasicBlock(0x130, 0x132, [ins(0x130, "add", "ebx, eax"), ins(0x131, "ret", "", "return")], [], ".text"),
    ]
    rows = lift_blocks(blocks, bits=64)
    result = construct_ssa(blocks, rows, 0x100)
    assert result["summary"]["phi_count"] == 1
    phi = next(op for op in result["operations"] if op["opcode"] == "phi")
    assert set(phi["incoming"].values()) == {"RAX_2", "RAX_3"}
    binary = next(op for op in result["operations"] if op["opcode"] == "binary")
    assert binary["inputs_ssa"][1] == phi["destination_ssa"]
    assert binary["expression_ssa"] == f"(RBX_0 + {phi['destination_ssa']})"
    block_map = {b["start"]: b for b in result["blocks"]}
    assert block_map["0x130"]["immediate_dominator"] == "0x100"
    assert result["summary"]["dominators_computed"] is True


def test_ssa_preserves_phi_operands_by_predecessor():
    blocks = [
        BasicBlock(0x200, 0x202, [ins(0x200, "mov", "eax, 1"), ins(0x201, "jne", "0x220", "conditional", 0x220)], [0x210, 0x220], ".text"),
        BasicBlock(0x210, 0x212, [ins(0x210, "mov", "eax, 2"), ins(0x211, "jmp", "0x230", "jump", 0x230)], [0x230], ".text"),
        BasicBlock(0x220, 0x222, [ins(0x220, "mov", "eax, 3"), ins(0x221, "jmp", "0x230", "jump", 0x230)], [0x230], ".text"),
        BasicBlock(0x230, 0x231, [ins(0x230, "ret", "", "return")], [], ".text"),
    ]
    rows = lift_blocks(blocks, bits=64)
    result = construct_ssa(blocks, rows, 0x200)
    phi = next(op for op in result["operations"] if op["opcode"] == "phi")
    assert phi["incoming"] == {"0x210": "RAX_2", "0x220": "RAX_3"}


def test_ssa_propagates_types_to_versions_and_phi_inputs():
    blocks = [
        BasicBlock(0x240, 0x242, [ins(0x240, "mov", "eax, 1"), ins(0x241, "jne", "0x260", "conditional", 0x260)], [0x250, 0x260], ".text"),
        BasicBlock(0x250, 0x252, [ins(0x250, "mov", "eax, 2"), ins(0x251, "jmp", "0x270", "jump", 0x270)], [0x270], ".text"),
        BasicBlock(0x260, 0x262, [ins(0x260, "mov", "eax, 3"), ins(0x261, "jmp", "0x270", "jump", 0x270)], [0x270], ".text"),
        BasicBlock(0x270, 0x272, [ins(0x270, "add", "ebx, eax"), ins(0x271, "ret", "", "return")], [], ".text"),
    ]
    result = construct_ssa(blocks, lift_blocks(blocks, bits=64), 0x240)
    phi = next(op for op in result["operations"] if op["opcode"] == "phi")
    assert phi["destination_type_ssa"] == "uint32_t"
    assert phi["input_types_ssa"] == ["uint32_t", "uint32_t"]
    assert result["summary"]["typed_version_count"] >= 3


def test_flags_are_explicit_ir_def_use_for_cmp_and_branch():
    blocks = [
        BasicBlock(0x300, 0x302, [ins(0x300, "cmp", "eax, 0"), ins(0x301, "je", "0x310", "conditional", 0x310)], [0x310], ".text"),
        BasicBlock(0x310, 0x311, [ins(0x310, "ret", "", "return")], [], ".text"),
    ]
    rows = lift_blocks(blocks, bits=64)
    cmp_row = rows[0]
    branch_row = rows[1]
    assert cmp_row["defs"] == ["FLAGS"]
    assert branch_row["inputs"] == ["FLAGS"]
    result = construct_ssa(blocks, rows, 0x300)
    branch_ssa = next(op for op in result["operations"] if op["opcode"] == "conditional_branch")
    assert branch_ssa["inputs_ssa"] == ["FLAGS_1"]


def test_x64_32bit_register_write_marks_upper_bits_zeroed():
    row = lift_instruction(ins(0x400, "mov", "eax, 5"), bits=64, block_start=0x400)
    assert row["zeroes_upper_bits"] is True


def test_implicit_wide_multiply_defines_rax_and_rdx():
    row = lift_instruction(ins(0x500, "mul", "ecx"), bits=64, block_start=0x500)
    assert row["opcode"] == "wide_multiply"
    assert row["defs"] == ["RAX", "RDX", "FLAGS"]
    assert row["inputs"] == ["RAX", "RCX"]


def test_division_uses_rax_rdx_and_defines_quotient_remainder():
    row = lift_instruction(ins(0x501, "idiv", "ecx"), bits=64, block_start=0x501)
    assert row["opcode"] == "divide"
    assert row["inputs"] == ["RDX", "RAX", "RCX"]
    assert row["defs"] == ["RAX", "RDX"]


def test_stack_and_exchange_effects_are_explicit():
    push = lift_instruction(ins(0x502, "push", "rax"), bits=64, block_start=0x502)
    pop = lift_instruction(ins(0x503, "pop", "rbx"), bits=64, block_start=0x503)
    exchange = lift_instruction(ins(0x504, "xchg", "rax, rbx"), bits=64, block_start=0x504)
    assert push["defs"] == ["RSP"]
    assert "RSP" in push["inputs"]
    assert pop["defs"] == ["RSP", "RBX"]
    assert pop["inputs"] == ["RSP"]
    assert exchange["defs"] == ["RAX", "RBX"]


def test_atomic_read_modify_write_instructions_have_explicit_effects():
    memory_exchange = lift_instruction(ins(0x50A, "xchg", "dword ptr [rax], ecx"), bits=64)
    exchange_add = lift_instruction(ins(0x50B, "xadd", "dword ptr [rax], ecx"), bits=64)
    compare_exchange = lift_instruction(ins(0x50C, "cmpxchg", "dword ptr [rax], ecx"), bits=64)
    assert memory_exchange["opcode"] == "atomic_exchange"
    assert memory_exchange["defs"] == ["MEM[RAX]", "RCX"]
    assert exchange_add["opcode"] == "exchange_add"
    assert exchange_add["defs"] == ["MEM[RAX]", "RCX", "FLAGS"]
    assert compare_exchange["opcode"] == "atomic_compare_exchange"
    assert compare_exchange["defs"] == ["MEM[RAX]", "RAX", "FLAGS"]


def test_byte_swap_has_explicit_register_effect():
    row = lift_instruction(ins(0x50D, "bswap", "eax"), bits=64)
    assert row["opcode"] == "byte_swap"
    assert row["destination"] == "RAX"
    assert row["defs"] == ["RAX"]
    assert row["width"] == 32


def test_rotate_through_carry_reads_and_writes_flags():
    row = lift_instruction(ins(0x50E, "rcl", "al, 1"), bits=64)
    assert row["opcode"] == "rotate"
    assert row["inputs"] == ["RAX", "const:1", "FLAGS"]
    assert row["defs"] == ["RAX", "FLAGS"]


def test_double_precision_shifts_have_typed_ir_effects():
    row = lift_instruction(ins(0x50F, "shld", "eax, ebx, cl"), bits=64)
    assert row["opcode"] == "double_shift"
    assert row["destination"] == "RAX"
    assert row["inputs"] == ["RAX", "RBX", "RCX"]
    assert row["defs"] == ["RAX", "FLAGS"]
    assert row["width"] == 32
    assert row["count_mask"] == 0x1F
    assert row["preserves_flags_when_count_zero"] is True
    assert row["result_undefined_when_count_exceeds_width"] is True


def test_rep_string_instruction_exposes_implicit_memory_and_register_effects():
    row = lift_instruction(ins(0x50F, "rep movsb"), bits=64)
    assert row["opcode"] == "string_move"
    assert row["repeat_prefix"] == "rep"
    assert row["width"] == 8
    assert row["inputs"] == ["MEM[RSI]", "RSI", "RDI", "DF", "RCX"]
    assert row["defs"] == ["MEM[RDI]", "RSI", "RDI", "RCX"]


def test_repe_string_compare_records_condition_flag_dependency():
    row = lift_instruction(ins(0x510, "repe cmpsd"), bits=64)
    assert row["opcode"] == "string_compare"
    assert row["repeat_prefix"] == "repe"
    assert row["repeat_condition"] == "repe"
    assert "FLAGS" in row["inputs"]
    assert "FLAGS" in row["defs"]


def test_vector_lane_extract_and_insert_have_explicit_ir_effects():
    extract = lift_instruction(ins(0x513, "pextrw", "eax, xmm0, 2"), bits=64)
    insert = lift_instruction(ins(0x514, "vpinsrd", "xmm1, xmm2, eax, 1"), bits=64)
    assert extract["opcode"] == "vector_extract"
    assert extract["destination"] == "RAX"
    assert extract["lane_width"] == 16
    assert extract["defs"] == ["RAX"]
    assert insert["opcode"] == "vector_insert"
    assert insert["destination"] == "XMM1"
    assert insert["lane_width"] == 32
    assert insert["defs"] == ["XMM1"]


def test_direction_flag_instructions_have_explicit_ir_effects():
    clear = lift_instruction(ins(0x511, "cld"), bits=64)
    set_flag = lift_instruction(ins(0x512, "std"), bits=64)
    assert clear["opcode"] == "direction_flag"
    assert clear["expression"] == "DF = 0"
    assert set_flag["expression"] == "DF = 1"
    assert clear["defs"] == ["DF"]


def test_carry_arithmetic_reads_flags():
    row = lift_instruction(ins(0x505, "adc", "eax, ecx"), bits=64, block_start=0x505)
    assert row["inputs"] == ["RAX", "RCX", "FLAGS"]


def test_bit_test_and_sign_extension_have_explicit_effects():
    bit = lift_instruction(ins(0x508, "bts", "eax, ecx"), bits=64, block_start=0x508)
    extend = lift_instruction(ins(0x509, "cqo", ""), bits=64, block_start=0x509)
    assert bit["opcode"] == "bit_test"
    assert bit["defs"] == ["RAX", "FLAGS"]
    assert extend["defs"] == ["RDX"]
    assert extend["inputs"] == ["RAX"]


def test_not_does_not_claim_to_modify_flags():
    row = lift_instruction(ins(0x50A, "not", "eax"), bits=64, block_start=0x50A)
    assert row["defs"] == ["RAX"]


def test_ir_propagates_pointer_and_boolean_type_hints():
    address = lift_instruction(ins(0x506, "lea", "rax, [rip + 0x20]"), bits=64, block_start=0x506)
    condition = lift_instruction(ins(0x507, "sete", "al"), bits=64, block_start=0x507)
    assert address["destination_type"] == "uint64_t"
    assert address["input_types"] == ["uintptr_t"]
    assert condition["destination_type"] == "bool"


def test_fence_and_spin_wait_instructions_have_explicit_ir_effects():
    full = lift_instruction(ins(0x515, "mfence"), bits=64)
    load = lift_instruction(ins(0x516, "lfence"), bits=64)
    store = lift_instruction(ins(0x517, "sfence"), bits=64)
    wait = lift_instruction(ins(0x518, "pause"), bits=64)
    assert full["opcode"] == "memory_fence" and full["defs"] == ["MEMORY_ORDER"]
    assert full["ordering"] == "full"
    assert load["ordering"] == "load_and_execution"
    assert store["ordering"] == "store"
    assert wait["opcode"] == "spin_wait_hint" and wait["defs"] == []


def test_scalar_sse_arithmetic_has_explicit_precision_and_lane_effects():
    legacy = lift_instruction(ins(0x519, "addss", "xmm0, xmm1"), bits=64)
    vex = lift_instruction(ins(0x51A, "vsubsd", "xmm0, xmm1, xmm2"), bits=64)
    assert legacy["opcode"] == "scalar_float_binary"
    assert legacy["precision"] == "single" and legacy["lane_width"] == 32
    assert legacy["defs"] == ["XMM0"]
    assert vex["precision"] == "double" and vex["lane_width"] == 64
    assert vex["inputs"] == ["XMM1", "XMM2"]
    assert vex["preserves_upper_scalar_lanes"] is True


def test_scalar_sse_minmax_ir_documents_nan_and_equal_selection():
    row = lift_instruction(ins(0x51B, "vminss", "xmm0, xmm1, xmm2"), bits=64)
    assert row["opcode"] == "scalar_float_minmax"
    assert row["inputs"] == ["XMM1", "XMM2"]
    assert row["unordered_or_equal_selects_second_source"] is True


def test_scalar_sse_conversions_describe_signedness_and_rounding():
    to_float = lift_instruction(ins(0x51C, "vcvtsi2ss", "xmm0, xmm1, eax"), bits=64)
    to_int = lift_instruction(ins(0x51D, "cvttsd2si", "rax, xmm2"), bits=64)
    assert to_float["opcode"] == "scalar_int_to_float"
    assert to_float["constant_fold_requires_exact_result"] is True
    assert to_int["opcode"] == "scalar_float_to_int"
    assert to_int["rounding"] == "truncate_toward_zero"
    assert to_int["precision"] == "double"


def test_scalar_sse_round_ir_exposes_mxcsr_and_immediate_control():
    legacy = lift_instruction(ins(0x51E, "roundss", "xmm0, xmm1, 5"), bits=64)
    vex = lift_instruction(ins(0x51F, "vroundsd", "xmm0, xmm1, xmm2, 6"), bits=64)
    assert legacy["opcode"] == "scalar_float_round"
    assert legacy["rounding_control"] == "immediate[1:0]"
    assert legacy["mxcsr_rounding_when_immediate_bit_2_is_clear"] is True
    assert legacy["constant_fold_requires_explicit_rounding_mode"] is True
    assert vex["precision"] == "double" and vex["lane_width"] == 64
    assert vex["inputs"] == ["XMM1", "XMM2", "const:6"]


def test_scalar_float_precision_conversion_ir_records_exactness():
    widen = lift_instruction(ins(0x51F1, "cvtss2sd", "xmm0, xmm1"), bits=64)
    narrow = lift_instruction(ins(0x51F2, "vcvtsd2ss", "xmm0, xmm1, xmm2"), bits=64)
    assert widen["opcode"] == "scalar_float_convert"
    assert widen["source_precision"] == "single"
    assert widen["target_precision"] == "double"
    assert widen["constant_fold_requires_exact_result"] is True
    assert narrow["source_precision"] == "double"
    assert narrow["target_precision"] == "single"
    assert narrow["mxcsr_rounding_needed_for_inexact_narrowing"] is True
    assert narrow["inputs"] == ["XMM1", "XMM2"]


def test_packed_float_arithmetic_ir_records_lane_and_exception_scope():
    legacy = lift_instruction(ins(0x51F3, "addps", "xmm0, xmm1"), bits=64)
    vex = lift_instruction(ins(0x51F4, "vdivpd", "ymm0, ymm1, ymm2"), bits=64)
    assert legacy["opcode"] == "vector_float_binary"
    assert legacy["precision"] == "single" and legacy["lane_width"] == 32
    assert legacy["lane_wise"] is True
    assert legacy["constant_fold_requires_exact_finite_result"] is True
    assert vex["precision"] == "double" and vex["lane_width"] == 64
    assert vex["inputs"] == ["YMM1", "YMM2"]
    assert vex["exception_flags_modeled"] is False


def test_packed_float_compare_ir_records_predicate_and_lane_masks():
    legacy = lift_instruction(ins(0x51F5, "cmpps", "xmm0, xmm1, 4"), bits=64)
    vex = lift_instruction(ins(0x51F6, "vcmppd", "ymm0, ymm1, ymm2, 2"), bits=64)
    assert legacy["opcode"] == "vector_float_compare"
    assert legacy["lane_width"] == 32
    assert legacy["predicate_operand"] == "const:4"
    assert legacy["supported_predicate_range"] == [0, 7]
    assert legacy["comparison_result"] == "all_ones_or_zero_per_lane"
    assert vex["precision"] == "double" and vex["lane_width"] == 64
    assert vex["supported_predicate_range"] == [0, 31]
    assert vex["nan_ordering"] == "explicit_ordered_unordered_predicates"


def test_packed_float_minmax_and_sqrt_ir_record_conservative_semantics():
    minimum = lift_instruction(ins(0x51F7, "vminpd", "ymm0, ymm1, ymm2"), bits=64)
    sqrt = lift_instruction(ins(0x51F8, "sqrtps", "xmm0, xmm1"), bits=64)
    assert minimum["opcode"] == "vector_float_minmax"
    assert minimum["lane_width"] == 64
    assert minimum["unordered_or_equal_selects_second_source"] is True
    assert minimum["preserves_operand_bit_patterns"] is True
    assert sqrt["opcode"] == "vector_float_sqrt"
    assert sqrt["lane_width"] == 32
    assert sqrt["constant_fold_requires_exact_nonnegative_result"] is True


def test_packed_integer_float_conversion_ir_records_rounding_rules():
    to_float = lift_instruction(ins(0x51F9, "cvtdq2ps", "xmm0, xmm1"), bits=64)
    rounded = lift_instruction(ins(0x51FA, "vcvtps2dq", "ymm0, ymm1"), bits=64)
    truncated = lift_instruction(ins(0x51FB, "cvttps2dq", "xmm0, xmm1"), bits=64)
    assert to_float["opcode"] == "vector_float_int_convert"
    assert to_float["source_element_type"] == "signed_int32"
    assert to_float["target_element_type"] == "float32"
    assert rounded["rounding"] == "mxcsr; fold integral values only"
    assert rounded["width"] == 256
    assert truncated["rounding"] == "truncate_toward_zero"
    assert truncated["out_of_range_behavior"] == "unknown"


def test_packed_float_precision_conversion_ir_records_lane_shape():
    widen = lift_instruction(ins(0x51FC, "vcvtps2pd", "ymm0, xmm1"), bits=64)
    narrow = lift_instruction(ins(0x51FD, "cvtpd2ps", "xmm0, xmm1"), bits=64)
    assert widen["opcode"] == "vector_float_convert"
    assert widen["source_precision"] == "single"
    assert widen["target_precision"] == "double"
    assert widen["source_lane_width"] == 32 and widen["destination_lane_width"] == 64
    assert widen["width"] == 256
    assert narrow["source_precision"] == "double"
    assert narrow["target_precision"] == "single"
    assert narrow["constant_fold_requires_exact_result"] is True


def test_packed_integer_multiply_ir_records_lane_and_result_selection():
    high = lift_instruction(ins(0x520, "pmulhw", "xmm0, xmm1"), bits=64)
    full = lift_instruction(ins(0x521, "vpmuldq", "ymm0, ymm1, ymm2"), bits=64)
    assert high["opcode"] == "vector_multiply"
    assert high["lane_width"] == 16
    assert high["result_selection"] == "high_half"
    assert high["signed"] is True
    assert full["lane_width"] == 64
    assert full["result_selection"] == "full_even_dword_products"


def test_packed_unpack_ir_records_lane_width_and_source_half():
    low = lift_instruction(ins(0x522, "punpcklbw", "xmm0, xmm1"), bits=64)
    high = lift_instruction(ins(0x523, "vpunpckhqdq", "ymm0, ymm1, ymm2"), bits=64)
    assert low["opcode"] == "vector_unpack"
    assert low["lane_width"] == 8 and low["source_half"] == "low"
    assert high["lane_width"] == 64 and high["source_half"] == "high"
    assert high["operates_per_128_bit_lane"] is True


def test_horizontal_sign_and_absolute_simd_have_typed_ir():
    horizontal = lift_instruction(ins(0x5231, "phaddsw", "xmm0, xmm1"), bits=64)
    sign = lift_instruction(ins(0x5232, "vpsignb", "ymm0, ymm1, ymm2"), bits=64)
    absolute = lift_instruction(ins(0x5233, "pabsw", "xmm0, xmm1"), bits=64)
    assert horizontal["opcode"] == "vector_horizontal_arithmetic"
    assert horizontal["lane_width"] == 16
    assert horizontal["signed_saturating"] is True
    assert horizontal["operates_per_128_bit_lane"] is True
    assert sign["opcode"] == "vector_sign"
    assert sign["lane_width"] == 8 and sign["signed_control_lane"] is True
    assert sign["zero_control_zeroes_lane"] is True
    assert absolute["opcode"] == "vector_absolute"
    assert absolute["lane_width"] == 16 and absolute["minimum_signed_lane_wraps"] is True


def test_immediate_vector_blends_have_lane_control_metadata():
    legacy = lift_instruction(ins(0x5234, "blendps", "xmm0, xmm1, 5"), bits=64)
    vex = lift_instruction(ins(0x5235, "vpblendd", "ymm0, ymm1, ymm2, 32"), bits=64)
    word = lift_instruction(ins(0x5236, "pblendw", "xmm0, xmm1, 129"), bits=64)
    assert legacy["opcode"] == "vector_blend"
    assert legacy["lane_width"] == 32 and legacy["selector_bit_count"] == 8
    assert legacy["inputs"] == ["XMM0", "XMM1", "const:5"]
    assert vex["inputs"] == ["YMM1", "YMM2", "const:32"]
    assert vex["lane_width"] == 32
    assert word["lane_width"] == 16
    assert word["selector_repeats_per_128_bit_lane"] is True


def test_variable_vector_blend_ir_records_mask_source():
    legacy = lift_instruction(ins(0x5237, "pblendvb", "xmm0, xmm1"), bits=64)
    vex = lift_instruction(ins(0x5238, "vpblendvb", "ymm0, ymm1, ymm2, ymm3"), bits=64)
    assert legacy["opcode"] == "vector_blend"
    assert legacy["lane_width"] == 8
    assert legacy["inputs"] == ["XMM0", "XMM1", "XMM0"]
    assert legacy["implicit_mask_register"] == "XMM0"
    assert legacy["control"] == "mask_byte_sign_bit"
    assert vex["inputs"] == ["YMM1", "YMM2", "YMM3"]
    assert vex["implicit_mask_register"] is None


def test_vector_align_and_test_flags_have_explicit_ir_effects():
    align = lift_instruction(ins(0x5239, "vpalignr", "ymm0, ymm1, ymm2, 8"), bits=64)
    test = lift_instruction(ins(0x523A, "ptest", "xmm0, xmm1"), bits=64)
    assert align["opcode"] == "vector_align"
    assert align["inputs"] == ["YMM1", "YMM2", "const:8"]
    assert align["lane_width"] == 128
    assert align["operates_per_128_bit_lane"] is True
    assert test["opcode"] == "vector_test_flags"
    assert test["defs"] == ["FLAGS"]
    assert test["flags_defined"] == ["CF", "ZF"]
    assert test["carry_flag"] == "no_bits_in_second_outside_first"
    assert test["zero_flag"] == "no_common_set_bits"


def test_movemask_and_saturating_pack_have_explicit_ir_effects():
    mask = lift_instruction(ins(0x524, "vpmovmskb", "eax, ymm1"), bits=64)
    pack = lift_instruction(ins(0x525, "packuswb", "xmm0, xmm1"), bits=64)
    assert mask["opcode"] == "vector_movemask"
    assert mask["element_width"] == 8 and mask["result_bit_count"] == 32
    assert pack["opcode"] == "vector_pack_saturate"
    assert pack["input_lane_width"] == 16 and pack["output_lane_width"] == 8
    assert pack["signed_output"] is False


def test_vector_shuffle_ir_describes_control_and_lane_scope():
    byte_shuffle = lift_instruction(ins(0x526, "vpshufb", "ymm0, ymm1, ymm2"), bits=64)
    immediate_shuffle = lift_instruction(ins(0x527, "shufps", "xmm0, xmm1, 0xb1"), bits=64)
    assert byte_shuffle["opcode"] == "vector_shuffle"
    assert byte_shuffle["shuffle_control"] == "byte_control"
    assert byte_shuffle["control_high_bit_zeroes_output"] is True
    assert byte_shuffle["operates_per_128_bit_lane"] is True
    assert immediate_shuffle["shuffle_control"] == "immediate"
    assert immediate_shuffle["lane_width"] == 32


def test_integer_count_instruction_ir_records_zero_case_and_flags():
    popcount = lift_instruction(ins(0x528, "popcnt", "eax, ecx"), bits=64)
    leading = lift_instruction(ins(0x529, "lzcnt", "rax, rdx"), bits=64)
    scan = lift_instruction(ins(0x52A, "bsf", "eax, ecx"), bits=64)
    assert popcount["opcode"] == "population_count"
    assert popcount["zero_flag"] == "source_is_zero"
    assert leading["opcode"] == "leading_zero_count"
    assert leading["zero_input_result"] == "operand_width"
    assert leading["carry_flag"] == "source_is_zero"
    assert scan["opcode"] == "bit_scan_forward"
    assert scan["zero_input_result"] == "undefined"


def test_bmi1_ir_exposes_bit_manipulation_and_flag_effects():
    and_not = lift_instruction(ins(0x52B, "andn", "eax, ecx, edx"), bits=64)
    isolate = lift_instruction(ins(0x52C, "blsi", "rax, rbx"), bits=64)
    assert and_not["opcode"] == "bitwise_and_not"
    assert and_not["defs"] == ["RAX", "FLAGS"]
    assert and_not["carry_flag"] == 0 and and_not["overflow_flag"] == 0
    assert isolate["opcode"] == "bit_manipulation"
    assert isolate["carry_flag"] == "source_is_zero"
    assert isolate["overflow_flag"] == 0


def test_pairwise_simd_ir_describes_lane_math_and_saturation():
    madd = lift_instruction(ins(0x52D, "vpmaddwd", "ymm0, ymm1, ymm2"), bits=64)
    madd_unsigned = lift_instruction(ins(0x52E, "pmaddubsw", "xmm0, xmm1"), bits=64)
    sad = lift_instruction(ins(0x52F, "psadbw", "xmm0, xmm1"), bits=64)
    assert madd["opcode"] == "vector_pairwise_arithmetic"
    assert madd["pairwise_behavior"] == "signed_pairwise_products_sum_wrap"
    assert madd_unsigned["pairwise_behavior"] == "unsigned_signed_pairwise_products_sum_saturate_signed"
    assert sad["pairwise_behavior"] == "sum_absolute_unsigned_byte_differences"
    assert sad["output_lane_width"] == 64


def test_x87_unary_sign_instructions_have_stack_effects():
    absolute = lift_instruction(ins(0x51E, "fabs"), bits=64)
    negate = lift_instruction(ins(0x51F, "fchs"), bits=64)
    assert absolute["opcode"] == "x87_unary"
    assert absolute["expression"] == "absolute_value(ST0)"
    assert absolute["defs"] == ["X87_STACK", "ST0"]
    assert negate["expression"] == "negate_sign(ST0)"
