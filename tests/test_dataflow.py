import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decompiler_engine.models import Instruction, BasicBlock
from decompiler_engine.dataflow import (
    AbstractState, FunctionDataFlow, _fold_packed_integer, _fold_vector_movemask,
    _fold_vector_shuffle,
)


def ins(addr, mnemonic, operands="", *, target=None, ref=None, flow="normal"):
    return Instruction(addr, 1, "90", mnemonic, operands, ".text", target, ref, flow)


def block(start, instructions, successors=None):
    return BasicBlock(start, instructions[-1].end, instructions, successors or [], ".text")


def test_constant_propagation_through_stack_slot_and_multiply():
    b = block(0x1000, [
        ins(0x1000, "mov", "dword ptr [rbp - 4], 6"),
        ins(0x1001, "mov", "eax, dword ptr [rbp - 4]"),
        ins(0x1002, "imul", "eax, eax"),
        ins(0x1003, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(64, {}, {}, {})
    rows, summary = flow.analyze(0x1000, [b])
    imul_row = next(r for r in rows if r["address"] == "0x1002")
    assert imul_row["write"]["value"] == "0x24"
    assert summary["rows"] == 4


def test_unmodeled_register_write_invalidates_stale_value():
    b = block(0x1010, [
        ins(0x1010, "mov", "eax, 7"),
        ins(0x1011, "movzx", "eax, byte ptr [rbx]"),
        ins(0x1012, "mov", "ecx, eax"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1010, [b])
    opaque_write = next(row for row in rows if row["address"] == "0x1011")
    following_copy = next(row for row in rows if row["address"] == "0x1012")
    assert opaque_write["write"] == {"target": "rax", "value": "?"}
    assert following_copy["write"]["value"] == "?"


def test_unmodeled_memory_write_invalidates_saved_memory_facts():
    b = block(0x1020, [
        ins(0x1020, "mov", "dword ptr [rax], 7"),
        ins(0x1021, "xsave", "[rax]"),
        ins(0x1022, "mov", "ecx, dword ptr [rax]"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1020, [b])
    memory_write = next(row for row in rows if row["address"] == "0x1021")
    following_load = next(row for row in rows if row["address"] == "0x1022")
    assert {"target": "memory[*]", "value": "?"} in memory_write["additional_writes"]
    assert following_load["write"]["value"] != "0x7"


def test_movzx_movsx_and_movsxd_extend_known_values_at_source_width():
    b = block(0x1030, [
        ins(0x1030, "mov", "al, 0xff"),
        ins(0x1031, "movzx", "ecx, al"),
        ins(0x1032, "movsx", "edx, al"),
        ins(0x1033, "mov", "eax, 0xffffffff"),
        ins(0x1034, "movsxd", "r8, eax"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1030, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x1031"]["write"]["value"] == "0xff"
    assert by_address["0x1032"]["write"]["value"] == "0xffffffff"
    assert by_address["0x1034"]["write"]["value"] == "0xffffffffffffffff"


def test_x86_push_arguments_are_reported_in_calling_order_then_cleaned():
    b = block(0x1040, [
        ins(0x1040, "mov", "eax, 0x2000"),
        ins(0x1041, "push", "eax"),
        ins(0x1042, "mov", "eax, 0x3000"),
        ins(0x1043, "push", "eax"),
        ins(0x1044, "call", "0x1000", target=0x1000, flow="call"),
        ins(0x1045, "add", "esp, 8"),
        ins(0x1046, "call", "0x1000", target=0x1000, flow="call"),
    ])
    rows, _ = FunctionDataFlow(32, {}, {}, {}).analyze(0x1040, [b])
    calls = [row for row in rows if row["mnemonic"] == "call"]
    assert calls[0]["call_argument_values"] == ["0x3000", "0x2000"]
    assert calls[1]["call_argument_values"] == []


def test_x86_pushed_call_arguments_flow_across_agreeing_cfg_paths():
    entry = block(0x10A0, [
        ins(0x10A0, "push", "0x2000"),
        ins(0x10A1, "jne", "0x10B0", target=0x10B0, flow="conditional"),
    ], [0x10B0, 0x10A2])
    fallthrough = block(0x10A2, [ins(0x10A2, "jmp", "0x10C0", target=0x10C0, flow="jump")], [0x10C0])
    taken = block(0x10B0, [ins(0x10B0, "jmp", "0x10C0", target=0x10C0, flow="jump")], [0x10C0])
    join = block(0x10C0, [ins(0x10C0, "call", "0x2000", target=0x2000, flow="call")])
    rows, _ = FunctionDataFlow(32, {}, {}, {}).analyze(0x10A0, [entry, fallthrough, taken, join])
    call = next(row for row in rows if row["address"] == "0x10c0")
    assert call["call_argument_values"] == ["0x2000"]

    conflict_entry = block(0x10D0, [ins(0x10D0, "jne", "0x10E0", target=0x10E0, flow="conditional")], [0x10E0, 0x10D1])
    left = block(0x10D1, [
        ins(0x10D1, "push", "0x2000"),
        ins(0x10D2, "jmp", "0x10F0", target=0x10F0, flow="jump"),
    ], [0x10F0])
    right = block(0x10E0, [
        ins(0x10E0, "push", "0x3000"),
        ins(0x10E1, "jmp", "0x10F0", target=0x10F0, flow="jump"),
    ], [0x10F0])
    conflict_join = block(0x10F0, [ins(0x10F0, "call", "0x2000", target=0x2000, flow="call")])
    conflict_rows, _ = FunctionDataFlow(32, {}, {}, {}).analyze(
        0x10D0, [conflict_entry, left, right, conflict_join]
    )
    conflict_call = next(row for row in conflict_rows if row["address"] == "0x10f0")
    assert conflict_call["call_argument_values"] == []


def test_partial_register_reads_and_writes_preserve_disjoint_bits():
    b = block(0x1048, [
        ins(0x1048, "mov", "eax, 0x12345678"),
        ins(0x1049, "mov", "al, 0xab"),
        ins(0x104A, "mov", "ah, 0xcd"),
        ins(0x104B, "movzx", "ecx, al"),
        ins(0x104C, "movzx", "edx, ah"),
        ins(0x104D, "mov", "ax, 0x4321"),
        ins(0x104E, "mov", "ebx, eax"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1048, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x1049"]["write"]["value"] == "0xab"
    assert by_address["0x104a"]["write"]["value"] == "0xcd"
    assert by_address["0x104b"]["write"]["value"] == "0xab"
    assert by_address["0x104c"]["write"]["value"] == "0xcd"
    assert by_address["0x104e"]["write"]["value"] == "0x12344321"


def test_common_packed_integer_simd_operations_fold_concrete_vectors():
    b = block(0x1050, [
        ins(0x1050, "mov", "eax, 0xffffffff"),
        ins(0x1051, "movd", "xmm0, eax"),
        ins(0x1052, "paddd", "xmm0, xmm0"),
        ins(0x1053, "movd", "ecx, xmm0"),
        ins(0x1054, "vpxor", "ymm1, ymm1, ymm1"),
        ins(0x1055, "vpsubq", "ymm1, ymm1, ymm1"),
        ins(0x1056, "mov", "eax, 0x80000000"),
        ins(0x1057, "movd", "xmm2, eax"),
        ins(0x1058, "psrad", "xmm2, 1"),
        ins(0x1059, "movd", "edx, xmm2"),
        ins(0x105A, "pslld", "xmm2, 32"),
        ins(0x105B, "movd", "r8d, xmm2"),
        ins(0x105C, "mov", "eax, 0x7f7f7f7f"),
        ins(0x105D, "movd", "xmm3, eax"),
        ins(0x105E, "paddsb", "xmm3, xmm3"),
        ins(0x105F, "pcmpeqd", "xmm3, xmm3"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1050, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x1052"]["write"]["value"] == "0xfffffffe"
    assert by_address["0x1053"]["write"]["value"] == "0xfffffffe"
    assert by_address["0x1055"]["write"]["value"] == "0"
    assert by_address["0x1059"]["write"]["value"] == "0xc0000000"
    assert by_address["0x105b"]["write"]["value"] == "0"
    assert by_address["0x105e"]["write"]["value"] == "0x7f7f7f7f"
    assert by_address["0x105f"]["write"]["value"] == "0xffffffffffffffffffffffffffffffff"


def test_packed_integer_multiply_low_high_and_even_dword_products():
    b = block(0x1068, [
        ins(0x1068, "mov", "eax, 0x00020003"),
        ins(0x1069, "movd", "xmm0, eax"),
        ins(0x106A, "mov", "ebx, 0x00040005"),
        ins(0x106B, "movd", "xmm1, ebx"),
        ins(0x106C, "pmullw", "xmm0, xmm1"),
        ins(0x106D, "mov", "eax, 0x7fff8000"),
        ins(0x106E, "movd", "xmm2, eax"),
        ins(0x106F, "mov", "ebx, 2"),
        ins(0x1070, "movd", "xmm3, ebx"),
        ins(0x1071, "pmulhw", "xmm2, xmm3"),
        ins(0x1072, "mov", "eax, 0x8000ffff"),
        ins(0x1073, "movd", "xmm4, eax"),
        ins(0x1074, "pmulhuw", "xmm4, xmm3"),
        ins(0x1075, "mov", "eax, 0xffffffff"),
        ins(0x1076, "movd", "xmm5, eax"),
        ins(0x1077, "mov", "ebx, 3"),
        ins(0x1078, "movd", "xmm6, ebx"),
        ins(0x1079, "pmulld", "xmm5, xmm6"),
        ins(0x107A, "mov", "eax, 0xfffffffe"),
        ins(0x107B, "movd", "xmm7, eax"),
        ins(0x107C, "mov", "ebx, 99"),
        ins(0x107D, "pinsrd", "xmm7, ebx, 1"),
        ins(0x107E, "mov", "ebx, 3"),
        ins(0x107F, "pinsrd", "xmm7, ebx, 2"),
        ins(0x1080, "mov", "ebx, 88"),
        ins(0x1081, "pinsrd", "xmm7, ebx, 3"),
        ins(0x1082, "mov", "eax, 4"),
        ins(0x1083, "movd", "xmm8, eax"),
        ins(0x1084, "mov", "ebx, 77"),
        ins(0x1085, "pinsrd", "xmm8, ebx, 1"),
        ins(0x1086, "mov", "ebx, 5"),
        ins(0x1087, "pinsrd", "xmm8, ebx, 2"),
        ins(0x1088, "mov", "ebx, 66"),
        ins(0x1089, "pinsrd", "xmm8, ebx, 3"),
        ins(0x108A, "vpmuldq", "xmm9, xmm7, xmm8"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1068, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x106c"]["write"]["value"] == "0x8000f"
    assert by_address["0x1071"]["write"]["value"] == "0xffff"
    assert by_address["0x1074"]["write"]["value"] == "1"
    assert by_address["0x1079"]["write"]["value"] == "0xfffffffd"
    assert by_address["0x108a"]["write"]["value"] == "0xffffffffffffffff8"


def test_packed_unpack_interleaves_low_and_high_halves_per_128bit_lane():
    left = int("1112131415161718191a1b1c1d1e1f20", 16)
    right = int("2122232425262728292a2b2c2d2e2f30", 16)
    low = _fold_packed_integer("punpcklbw", hex(left), hex(right), 128)
    high = _fold_packed_integer("punpckhbw", hex(left), hex(right), 128)
    assert low == "0x29192a1a2b1b2c1c2d1d2e1e2f1f3020"
    assert high == "0x21112212231324142515261627172818"

    left_wide = (left << 128) | left
    right_wide = (right << 128) | right
    vex = _fold_packed_integer("vpunpcklbw", hex(left_wide), hex(right_wide), 256)
    assert vex == hex(int(low, 16) | (int(low, 16) << 128))


def test_vector_movemask_extracts_sign_bits_and_packed_saturating_operations_fold():
    def lanes(values, width):
        return sum((value & ((1 << width) - 1)) << (index * width)
                   for index, value in enumerate(values))

    vectors = block(0x1090, [
        ins(0x1090, "mov", "eax, 0x80000000"),
        ins(0x1091, "movd", "xmm0, eax"),
        ins(0x1092, "mov", "ebx, 1"),
        ins(0x1093, "pinsrd", "xmm0, ebx, 1"),
        ins(0x1094, "mov", "ebx, 0xffffffff"),
        ins(0x1095, "pinsrd", "xmm0, ebx, 2"),
        ins(0x1096, "mov", "ebx, 0x80000000"),
        ins(0x1097, "pinsrd", "xmm0, ebx, 3"),
        ins(0x1098, "movmskps", "ecx, xmm0"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1090, [vectors])
    assert rows[-1]["write"]["value"] == "0xd"

    a = lanes([32767, -32768, 128, -129, 0, 1, 2, 3], 16)
    b = lanes([127, -128, 255, -255, 4, 5, 6, 7], 16)
    assert _fold_packed_integer("packsswb", hex(a), hex(b), 128) == "0x7060504807f807f03020100807f807f"
    u = lanes([-1, -32768, 256, 255, 128, 0, 1, 2], 16)
    v = lanes([3, 4, 5, 6, 7, 8, 9, 10], 16)
    assert _fold_packed_integer("packuswb", hex(u), hex(v), 128) == "0xa0908070605040302010080ffff0000"
    mask_source = int("80000000800000008000000080000000", 16)
    assert _fold_vector_movemask("movmskps", hex(mask_source), 128) == "0xf"
    assert _fold_vector_movemask("pmovmskb", hex(mask_source), 128) == "0x8888"


def test_vector_shuffle_folds_immediates_and_byte_control_with_lane_boundaries():
    source_a = sum(index << (index * 32) for index in range(4))
    source_b = sum((index + 4) << (index * 32) for index in range(4))
    assert _fold_vector_shuffle("pshufd", hex(source_a), "0", "0x1b", 128) == "0x10000000200000003"
    assert _fold_vector_shuffle("shufps", hex(source_a), hex(source_b), "0xb1", 128) == "0x6000000070000000000000001"
    double_a = 10 | (11 << 64)
    double_b = 12 | (13 << 64)
    assert _fold_vector_shuffle("shufpd", hex(double_a), hex(double_b), "2", 128) == "0xd000000000000000a"
    byte_source = sum(index << (index * 8) for index in range(16))
    controls = 15 | (0 << 8) | (0x80 << 16) | (0x8F << 24)
    assert _fold_vector_shuffle("pshufb", hex(byte_source), hex(controls), "0", 128) == "0xf"

    wide_a = (source_a << 128) | source_a
    wide_b = (source_b << 128) | source_b
    wide = _fold_vector_shuffle("vshufps", hex(wide_a), hex(wide_b), "0xb1", 256)
    one_lane = int(_fold_vector_shuffle("shufps", hex(source_a), hex(source_b), "0xb1", 128), 16)
    assert wide == hex(one_lane | (one_lane << 128))

    program = block(0x10B0, [
        ins(0x10B0, "mov", "eax, 1"),
        ins(0x10B1, "movd", "xmm0, eax"),
        ins(0x10B2, "mov", "ebx, 2"),
        ins(0x10B3, "pinsrd", "xmm0, ebx, 1"),
        ins(0x10B4, "mov", "ebx, 3"),
        ins(0x10B5, "pinsrd", "xmm0, ebx, 2"),
        ins(0x10B6, "mov", "ebx, 4"),
        ins(0x10B7, "pinsrd", "xmm0, ebx, 3"),
        ins(0x10B8, "pshufd", "xmm1, xmm0, 0x1b"),
        ins(0x10B9, "pextrd", "ecx, xmm1, 0"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x10B0, [program])
    assert rows[-1]["write"]["value"] == "4"


def test_population_and_zero_count_instructions_fold_values_and_flags():
    b = block(0x10C0, [
        ins(0x10C0, "mov", "eax, 0x80000001"),
        ins(0x10C1, "popcnt", "ecx, eax"),
        ins(0x10C2, "lzcnt", "edx, eax"),
        ins(0x10C3, "tzcnt", "ebx, eax"),
        ins(0x10C4, "bsf", "esi, eax"),
        ins(0x10C5, "bsr", "edi, eax"),
        ins(0x10C6, "mov", "eax, 0"),
        ins(0x10C7, "lzcnt", "ecx, eax"),
        ins(0x10C8, "jc", "0x10d0", target=0x10d0, flow="conditional"),
        ins(0x10C9, "tzcnt", "edx, eax"),
        ins(0x10CA, "jz", "0x10d0", target=0x10d0, flow="conditional"),
        ins(0x10CB, "bsf", "ebx, eax"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x10C0, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x10c1"]["write"]["value"] == "2"
    assert by_address["0x10c1"]["zero_flag_after"] is False
    assert by_address["0x10c2"]["write"]["value"] == "0"
    assert by_address["0x10c3"]["write"]["value"] == "0"
    assert by_address["0x10c4"]["write"]["value"] == "0"
    assert by_address["0x10c5"]["write"]["value"] == "0x1f"
    assert by_address["0x10c7"]["write"]["value"] == "0x20"
    assert by_address["0x10c7"]["carry_flag_after"] is True
    assert by_address["0x10c8"]["condition"] == "1"
    assert by_address["0x10c9"]["write"]["value"] == "0x20"
    assert by_address["0x10c9"]["zero_flag_after"] is False
    assert by_address["0x10ca"]["condition"] == "0"
    assert by_address["0x10cb"]["write"]["value"] == "?"


def test_bmi1_bit_manipulation_operations_fold_and_set_flags():
    b = block(0x10D0, [
        ins(0x10D0, "mov", "eax, 0x28"),
        ins(0x10D1, "blsi", "ecx, eax"),
        ins(0x10D2, "blsmsk", "edx, eax"),
        ins(0x10D3, "blsr", "ebx, eax"),
        ins(0x10D4, "andn", "esi, eax, 0x3f"),
        ins(0x10D5, "mov", "eax, 0"),
        ins(0x10D6, "blsi", "edi, eax"),
        ins(0x10D7, "jc", "0x10e0", target=0x10e0, flow="conditional"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x10D0, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x10d1"]["write"]["value"] == "8"
    assert by_address["0x10d2"]["write"]["value"] == "0xf"
    assert by_address["0x10d3"]["write"]["value"] == "0x20"
    assert by_address["0x10d4"]["write"]["value"] == "0x17"
    assert by_address["0x10d6"]["write"]["value"] == "0"
    assert by_address["0x10d6"]["carry_flag_after"] is True
    assert by_address["0x10d7"]["condition"] == "1"


def test_pairwise_simd_arithmetic_wraps_saturates_and_sums_byte_differences():
    def lanes(values, width):
        return sum((value & ((1 << width) - 1)) << (index * width)
                   for index, value in enumerate(values))

    all_words = lanes([-32768, -32768] + [0] * 6, 16)
    assert _fold_packed_integer("pmaddwd", hex(all_words), hex(all_words), 128) == "0x80000000"

    unsigned_bytes = lanes([255, 255, 1, 1] + [0] * 12, 8)
    signed_bytes = lanes([127, 127, 0x80, 0x80] + [0] * 12, 8)
    assert _fold_packed_integer("pmaddubsw", hex(unsigned_bytes), hex(signed_bytes), 128) == "0xff007fff"

    left = lanes(list(range(1, 17)), 8)
    right = 0
    assert _fold_packed_integer("psadbw", hex(left), hex(right), 128) == "0x640000000000000024"

    wide_left = (left << 128) | left
    wide_right = 0
    wide_result = _fold_packed_integer("vpsadbw", hex(wide_left), hex(wide_right), 256)
    assert wide_result == "0x64000000000000002400000000000000640000000000000024"


def test_horizontal_simd_sign_and_absolute_operations_fold_by_lane():
    def lanes(values, width):
        return sum((value & ((1 << width) - 1)) << (index * width)
                   for index, value in enumerate(values))

    left = lanes([1, 2, 3, 4, 5, 6, 7, 8], 16)
    right = lanes([10, 20, 30, 40, 50, 60, 70, 80], 16)
    assert _fold_packed_integer("phaddw", hex(left), hex(right), 128) == hex(
        lanes([3, 7, 11, 15, 30, 70, 110, 150], 16))

    saturating = lanes([32767, 1, -32768, -1, 0, 0, 0, 0], 16)
    assert _fold_packed_integer("phaddsw", hex(saturating), hex(saturating), 128) == hex(
        lanes([32767, -32768, 0, 0, 32767, -32768, 0, 0], 16))

    data = lanes([-5, 4, -128, 8, 0, 0, 0, 0, -2, 3, 0, 0, 0, 0, 0, 0], 8)
    control = lanes([-1, -1, 0, 1] + [0] * 12, 8)
    assert _fold_packed_integer("psignb", hex(data), hex(control), 128) == hex(
        lanes([5, -4, 0, 8] + [0] * 12, 8))
    words = lanes([-5, -32768, 7, -9, 0, 0, 0, 0], 16)
    assert _fold_packed_integer("pabsw", hex(words), "0", 128) == hex(
        lanes([5, -32768, 7, 9, 0, 0, 0, 0], 16))

    wide = (left << 128) | left
    assert _fold_packed_integer("vphaddw", hex(wide), hex(wide), 256) == hex(
        (lanes([3, 7, 11, 15, 3, 7, 11, 15], 16) << 128)
        | lanes([3, 7, 11, 15, 3, 7, 11, 15], 16))


def test_immediate_vector_blends_select_requested_source_lanes():
    def lanes(values, width):
        return sum((value & ((1 << width) - 1)) << (index * width)
                   for index, value in enumerate(values))

    a = lanes([1, 2, 3, 4], 32)
    b = lanes([10, 20, 30, 40], 32)
    assert _fold_vector_shuffle("blendps", hex(a), hex(b), "5", 128) == hex(
        lanes([10, 2, 30, 4], 32))
    words_a, words_b = lanes(list(range(8)), 16), lanes(list(range(10, 18)), 16)
    assert _fold_vector_shuffle("pblendw", hex(words_a), hex(words_b), "129", 128) == hex(
        lanes([10, 1, 2, 3, 4, 5, 6, 17], 16))

    wide_a, wide_b = lanes(list(range(8)), 32), lanes(list(range(20, 28)), 32)
    vex = block(0x10C0, [ins(0x10C0, "vpblendd", "ymm0, ymm1, ymm2, 32")])
    flow = FunctionDataFlow(64, {}, {}, {})
    _, rows = flow.transfer(vex, AbstractState(regs={
        "ymm1": hex(wide_a), "ymm2": hex(wide_b),
    }), 0x10C0, collect=True)
    assert rows[0]["write"]["value"] == hex(lanes([0, 1, 2, 3, 4, 25, 6, 7], 32))


def test_variable_byte_blends_use_implicit_or_explicit_mask_register():
    source_a = int.from_bytes(bytes(range(1, 17)), "little")
    source_b = int.from_bytes(bytes(range(101, 117)), "little")
    mask = int.from_bytes(bytes([0x80, 0, 0xFF, 0] * 4), "little")

    legacy = block(0x10D0, [ins(0x10D0, "pblendvb", "xmm1, xmm2")])
    flow = FunctionDataFlow(64, {}, {}, {})
    _, rows = flow.transfer(legacy, AbstractState(regs={
        "xmm0": hex(mask), "xmm1": hex(source_a), "xmm2": hex(source_b),
    }), 0x10D0, collect=True)
    expected_bytes = bytes(
        (101 + index if index % 4 in {0, 2} else 1 + index)
        for index in range(16)
    )
    expected = int.from_bytes(expected_bytes, "little")
    assert rows[0]["write"]["value"] == hex(expected)

    wide_mask = mask | (mask << 128)
    wide_a = source_a | (source_a << 128)
    wide_b = source_b | (source_b << 128)
    vex = block(0x10D1, [ins(0x10D1, "vpblendvb", "ymm0, ymm1, ymm2, ymm3")])
    _, rows = flow.transfer(vex, AbstractState(regs={
        "ymm1": hex(wide_a), "ymm2": hex(wide_b), "ymm3": hex(wide_mask),
    }), 0x10D1, collect=True)
    assert rows[0]["write"]["value"] == hex(expected | (expected << 128))


def test_palignr_aligns_inside_each_128_bit_lane_and_ptest_sets_flags():
    dest_bytes = bytes(range(1, 17))
    src_bytes = bytes(range(101, 117))
    dest = int.from_bytes(dest_bytes, "little")
    src = int.from_bytes(src_bytes, "little")
    aligned = _fold_vector_shuffle("palignr", hex(dest), hex(src), "8", 128)
    assert aligned == hex(int.from_bytes(src_bytes[8:] + dest_bytes[:8], "little"))
    assert _fold_vector_shuffle("palignr", hex(dest), hex(src), "32", 128) == "0"

    lane_a = dest | (dest << 128)
    lane_b = src | (src << 128)
    wide = _fold_vector_shuffle("vpalignr", hex(lane_a), hex(lane_b), "8", 256)
    expected_lane = int.from_bytes(src_bytes[8:] + dest_bytes[:8], "little")
    assert wide == hex(expected_lane | (expected_lane << 128))

    ptest = block(0x10E0, [ins(0x10E0, "ptest", "xmm0, xmm1")])
    flow = FunctionDataFlow(64, {}, {}, {})
    _, rows = flow.transfer(ptest, AbstractState(regs={"xmm0": "0x3", "xmm1": "0x1"}),
                            0x10E0, collect=True)
    assert rows[0]["zero_flag_after"] is False
    assert rows[0]["carry_flag_after"] is True
    _, rows = flow.transfer(ptest, AbstractState(regs={"xmm0": "0x1", "xmm1": "0x2"}),
                            0x10E0, collect=True)
    assert rows[0]["zero_flag_after"] is True
    assert rows[0]["carry_flag_after"] is False


def test_scalar_round_instructions_obey_immediate_modes_and_mxcsr_unknown():
    flow = FunctionDataFlow(64, {}, {}, {})
    base = 0x112233445566778899AABBCC
    expected_modes = {4: 0x40000000, 5: 0x3F800000,
                      6: 0x40000000, 7: 0x3F800000}
    for immediate, rounded in expected_modes.items():
        rounding = block(0x10F0, [ins(0x10F0, "roundss", f"xmm0, xmm1, {immediate}")])
        _, rows = flow.transfer(rounding, AbstractState(regs={
            "xmm0": hex(base), "xmm1": "0x3fe00000",
        }), 0x10F0, collect=True)
        assert rows[0]["write"]["value"] == hex((base & ~0xFFFFFFFF) | rounded)

    negative = block(0x10F1, [ins(0x10F1, "roundss", "xmm0, xmm1, 6")])
    _, rows = flow.transfer(negative, AbstractState(regs={
        "xmm0": hex(base), "xmm1": "0xbe99999a",
    }), 0x10F1, collect=True)
    assert rows[0]["write"]["value"] == hex((base & ~0xFFFFFFFF) | 0x80000000)

    mxcsr = block(0x10F2, [ins(0x10F2, "roundss", "xmm0, xmm1, 0")])
    _, rows = flow.transfer(mxcsr, AbstractState(regs={
        "xmm0": hex(base), "xmm1": "0x3fc00000",
    }), 0x10F2, collect=True)
    assert rows[0]["write"]["value"] == "?"

    vex = block(0x10F3, [ins(0x10F3, "vroundss", "xmm0, xmm1, xmm2, 5")])
    _, rows = flow.transfer(vex, AbstractState(regs={
        "xmm1": hex(base), "xmm2": "0x3fe00000",
    }), 0x10F3, collect=True)
    assert rows[0]["write"]["value"] == hex((base & ~0xFFFFFFFF) | 0x3F800000)


def test_scalar_float_precision_conversions_fold_only_exact_narrowing():
    flow = FunctionDataFlow(64, {}, {}, {})
    base = 0x112233445566778899AABBCC
    widen = block(0x1100, [ins(0x1100, "cvtss2sd", "xmm0, xmm1")])
    _, rows = flow.transfer(widen, AbstractState(regs={
        "xmm0": hex(base), "xmm1": "0x3fc00000",
    }), 0x1100, collect=True)
    assert rows[0]["write"]["value"] == hex((base & ~0xFFFFFFFFFFFFFFFF) | 0x3FF8000000000000)

    narrow = block(0x1101, [ins(0x1101, "cvtsd2ss", "xmm0, xmm1")])
    _, rows = flow.transfer(narrow, AbstractState(regs={
        "xmm0": hex(base), "xmm1": "0x3ff8000000000000",
    }), 0x1101, collect=True)
    assert rows[0]["write"]["value"] == hex((base & ~0xFFFFFFFF) | 0x3FC00000)
    _, rows = flow.transfer(narrow, AbstractState(regs={
        "xmm0": hex(base), "xmm1": "0x3ff199999999999a",
    }), 0x1101, collect=True)
    assert rows[0]["write"]["value"] == "?"

    vex = block(0x1102, [ins(0x1102, "vcvtss2sd", "xmm0, xmm1, xmm2")])
    _, rows = flow.transfer(vex, AbstractState(regs={
        "xmm1": hex(base), "xmm2": "0x80000000",
    }), 0x1102, collect=True)
    assert rows[0]["write"]["value"] == hex((base & ~0xFFFFFFFFFFFFFFFF) | 0x8000000000000000)


def test_packed_float_arithmetic_folds_only_exact_finite_lanes():
    def lanes(values, width):
        return sum((value & ((1 << width) - 1)) << (index * width)
                   for index, value in enumerate(values))

    flow = FunctionDataFlow(64, {}, {}, {})
    single_a = lanes([0x3FC00000] * 4, 32)  # 1.5
    single_b = lanes([0x40100000] * 4, 32)  # 2.25
    single_sum = lanes([0x40700000] * 4, 32)  # 3.75
    add = block(0x1110, [ins(0x1110, "addps", "xmm0, xmm1")])
    _, rows = flow.transfer(add, AbstractState(regs={
        "xmm0": hex(single_a), "xmm1": hex(single_b),
    }), 0x1110, collect=True)
    assert rows[0]["write"]["value"] == hex(single_sum)

    wide_a, wide_b = lanes([0x3FC00000] * 8, 32), lanes([0x40100000] * 8, 32)
    vex = block(0x1111, [ins(0x1111, "vaddps", "ymm0, ymm1, ymm2")])
    _, rows = flow.transfer(vex, AbstractState(regs={
        "ymm1": hex(wide_a), "ymm2": hex(wide_b),
    }), 0x1111, collect=True)
    assert rows[0]["write"]["value"] == hex(lanes([0x40700000] * 8, 32))

    double_a = lanes([0x3FF8000000000000] * 2, 64)
    double_b = lanes([0x4000000000000000] * 2, 64)
    double_product = lanes([0x4008000000000000] * 2, 64)
    mul = block(0x1112, [ins(0x1112, "mulpd", "xmm0, xmm1")])
    _, rows = flow.transfer(mul, AbstractState(regs={
        "xmm0": hex(double_a), "xmm1": hex(double_b),
    }), 0x1112, collect=True)
    assert rows[0]["write"]["value"] == hex(double_product)

    inexact = block(0x1113, [ins(0x1113, "addps", "xmm0, xmm1")])
    _, rows = flow.transfer(inexact, AbstractState(regs={
        "xmm0": hex(lanes([0x3F800000] * 4, 32)),
        "xmm1": hex(lanes([0x33800000] * 4, 32)),
    }), 0x1113, collect=True)
    assert rows[0]["write"]["value"] == "?"


def test_packed_float_comparisons_produce_ordered_unordered_lane_masks():
    def lanes(values, width):
        return sum((value & ((1 << width) - 1)) << (index * width)
                   for index, value in enumerate(values))

    left = lanes([0x3F800000, 0x7FC00000, 0x40000000, 0x40400000], 32)
    right = lanes([0x40000000, 0x40800000, 0x40000000, 0x7FC00000], 32)
    flow = FunctionDataFlow(64, {}, {}, {})
    less = block(0x1120, [ins(0x1120, "cmpps", "xmm0, xmm1, 1")])
    _, rows = flow.transfer(less, AbstractState(regs={
        "xmm0": hex(left), "xmm1": hex(right),
    }), 0x1120, collect=True)
    assert rows[0]["write"]["value"] == hex(lanes([0xFFFFFFFF, 0, 0, 0], 32))

    unordered_not_equal = block(0x1121, [ins(0x1121, "cmpps", "xmm0, xmm1, 4")])
    _, rows = flow.transfer(unordered_not_equal, AbstractState(regs={
        "xmm0": hex(left), "xmm1": hex(right),
    }), 0x1121, collect=True)
    assert rows[0]["write"]["value"] == hex(lanes([0xFFFFFFFF, 0xFFFFFFFF, 0, 0xFFFFFFFF], 32))

    vex = block(0x1122, [ins(0x1122, "vcmppd", "ymm0, ymm1, ymm2, 2")])
    doubles_left = lanes([0x3FF0000000000000] * 4, 64)
    doubles_right = lanes([0x3FF0000000000000, 0x4000000000000000,
                           0x3FE0000000000000, 0x3FF0000000000000], 64)
    _, rows = flow.transfer(vex, AbstractState(regs={
        "ymm1": hex(doubles_left), "ymm2": hex(doubles_right),
    }), 0x1122, collect=True)
    assert rows[0]["write"]["value"] == hex(lanes([0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
                                                     0, 0xFFFFFFFFFFFFFFFF], 64))

    unsupported = block(0x1123, [ins(0x1123, "cmpps", "xmm0, xmm1, 8")])
    _, rows = flow.transfer(unsupported, AbstractState(regs={
        "xmm0": hex(left), "xmm1": hex(right),
    }), 0x1123, collect=True)
    assert rows[0]["write"]["value"] == "?"

    relationships = ["gt", "lt", "eq", "unordered", "unordered", "unordered", "unordered", "unordered"]
    predicates = [
        ["eq"], ["lt"], ["lt", "eq"], ["unordered"],
        ["unordered", "lt", "gt"], ["unordered", "gt", "eq"],
        ["unordered", "gt"], ["gt", "lt", "eq"],
        ["unordered", "eq"], ["unordered", "lt"],
        ["unordered", "lt", "eq"], [], ["gt", "lt"],
        ["gt", "eq"], ["gt"], relationships,
        ["eq"], ["lt"], ["lt", "eq"], ["unordered"],
        ["unordered", "lt", "gt"], ["unordered", "gt", "eq"],
        ["unordered", "gt"], ["gt", "lt", "eq"],
        ["unordered", "eq"], ["unordered", "lt"],
        ["unordered", "lt", "eq"], [], ["gt", "lt"],
        ["gt", "eq"], ["gt"], relationships,
    ]
    left_wide = lanes([0x40000000, 0x3F800000, 0x3F800000, 0x7FC00000,
                       0x40000000, 0x3F800000, 0x3F800000, 0x7FC00000], 32)
    right_wide = lanes([0x3F800000, 0x40000000, 0x3F800000, 0x3F800000,
                        0x7FC00000, 0x7FC00000, 0x7FC00000, 0x7FC00000], 32)
    vex_predicate = block(0x1124, [ins(0x1124, "vcmpps", "ymm0, ymm1, ymm2, 0")])
    for predicate, accepted in enumerate(predicates):
        vex_predicate.instructions[0].operands = f"ymm0, ymm1, ymm2, {predicate}"
        _, rows = flow.transfer(vex_predicate, AbstractState(regs={
            "ymm1": hex(left_wide), "ymm2": hex(right_wide),
        }), 0x1124, collect=True)
        expected = lanes([
            0xFFFFFFFF if relation in accepted else 0 for relation in relationships
        ], 32)
        assert rows[0]["write"]["value"] == (hex(expected) if expected else "0"), predicate


def test_packed_float_minmax_preserves_selected_bits_and_sqrt_folds_exact_lanes():
    def lanes(values, width):
        return sum((value & ((1 << width) - 1)) << (index * width)
                   for index, value in enumerate(values))

    left = lanes([0x3F800000, 0x7FC12345, 0, 0x40400000], 32)
    right = lanes([0x40000000, 0x40800000, 0x80000000, 0x7FC54321], 32)
    flow = FunctionDataFlow(64, {}, {}, {})
    minimum = block(0x1130, [ins(0x1130, "minps", "xmm0, xmm1")])
    _, rows = flow.transfer(minimum, AbstractState(regs={
        "xmm0": hex(left), "xmm1": hex(right),
    }), 0x1130, collect=True)
    assert rows[0]["write"]["value"] == hex(lanes(
        [0x3F800000, 0x40800000, 0x80000000, 0x7FC54321], 32))

    wide_min = block(0x1131, [ins(0x1131, "vmaxps", "ymm0, ymm1, ymm2")])
    wide_left = lanes([0x3F800000] * 8, 32)
    wide_right = lanes([0x40000000] * 8, 32)
    _, rows = flow.transfer(wide_min, AbstractState(regs={
        "ymm1": hex(wide_left), "ymm2": hex(wide_right),
    }), 0x1131, collect=True)
    assert rows[0]["write"]["value"] == hex(wide_right)

    square_roots = block(0x1132, [ins(0x1132, "sqrtps", "xmm0, xmm1")])
    squares = lanes([0x40800000, 0x41100000, 0, 0x41800000], 32)  # 4, 9, 0, 16
    _, rows = flow.transfer(square_roots, AbstractState(regs={"xmm1": hex(squares)}),
                            0x1132, collect=True)
    assert rows[0]["write"]["value"] == hex(lanes(
        [0x40000000, 0x40400000, 0, 0x40800000], 32))

    irrational = block(0x1133, [ins(0x1133, "sqrtps", "xmm0, xmm1")])
    _, rows = flow.transfer(irrational, AbstractState(regs={
        "xmm1": hex(lanes([0x40400000] * 4, 32)),  # sqrt(3)
    }), 0x1133, collect=True)
    assert rows[0]["write"]["value"] == "?"


def test_packed_integer_float_conversions_obey_exact_and_truncation_rules():
    def lanes(values):
        return sum((value & 0xFFFFFFFF) << (index * 32)
                   for index, value in enumerate(values))

    flow = FunctionDataFlow(64, {}, {}, {})
    integer_to_float = block(0x1140, [ins(0x1140, "cvtdq2ps", "xmm0, xmm1")])
    integers = lanes([-2, 5, 16777216, 0])
    expected_floats = lanes([0xC0000000, 0x40A00000, 0x4B800000, 0])
    _, rows = flow.transfer(integer_to_float, AbstractState(regs={"xmm1": hex(integers)}),
                            0x1140, collect=True)
    assert rows[0]["write"]["value"] == hex(expected_floats)
    _, rows = flow.transfer(integer_to_float, AbstractState(regs={
        "xmm1": hex(lanes([16777217, 0, 0, 0])),
    }), 0x1140, collect=True)
    assert rows[0]["write"]["value"] == "?"

    floats = block(0x1141, [ins(0x1141, "cvtps2dq", "xmm0, xmm1")])
    integral_floats = lanes([0x3F800000, 0xC0000000, 0x40A00000, 0])
    _, rows = flow.transfer(floats, AbstractState(regs={"xmm1": hex(integral_floats)}),
                            0x1141, collect=True)
    assert rows[0]["write"]["value"] == hex(lanes([1, -2, 5, 0]))
    fractional = block(0x1142, [ins(0x1142, "cvtps2dq", "xmm0, xmm1")])
    _, rows = flow.transfer(fractional, AbstractState(regs={
        "xmm1": hex(lanes([0x3FE00000, 0xC0000000, 0, 0])),
    }), 0x1142, collect=True)
    assert rows[0]["write"]["value"] == "?"

    truncating = block(0x1143, [ins(0x1143, "cvttps2dq", "xmm0, xmm1")])
    values = lanes([0x3FE00000, 0xC0300000, 0x3F000000, 0xBF000000])
    _, rows = flow.transfer(truncating, AbstractState(regs={"xmm1": hex(values)}),
                            0x1143, collect=True)
    assert rows[0]["write"]["value"] == hex(lanes([1, -2, 0, 0]))


def test_packed_float_precision_conversions_handle_different_vector_widths():
    def lanes(values, width):
        return sum((value & ((1 << width) - 1)) << (index * width)
                   for index, value in enumerate(values))

    flow = FunctionDataFlow(64, {}, {}, {})
    widen = block(0x1150, [ins(0x1150, "cvtps2pd", "xmm0, xmm1")])
    singles = lanes([0x3FC00000, 0x40100000, 0, 0], 32)
    doubles = lanes([0x3FF8000000000000, 0x4002000000000000], 64)
    _, rows = flow.transfer(widen, AbstractState(regs={"xmm1": hex(singles)}),
                            0x1150, collect=True)
    assert rows[0]["write"]["value"] == hex(doubles)

    vex_widen = block(0x1151, [ins(0x1151, "vcvtps2pd", "ymm0, xmm1")])
    four_singles = lanes([0x3FC00000, 0x40100000, 0xC0000000, 0], 32)
    four_doubles = lanes([0x3FF8000000000000, 0x4002000000000000,
                          0xC000000000000000, 0], 64)
    _, rows = flow.transfer(vex_widen, AbstractState(regs={"xmm1": hex(four_singles)}),
                            0x1151, collect=True)
    assert rows[0]["write"]["value"] == hex(four_doubles)

    narrow = block(0x1152, [ins(0x1152, "cvtpd2ps", "xmm0, xmm1")])
    _, rows = flow.transfer(narrow, AbstractState(regs={"xmm1": hex(doubles)}),
                            0x1152, collect=True)
    assert rows[0]["write"]["value"] == hex(lanes([0x3FC00000, 0x40100000], 32))
    inexact = lanes([0x3FF199999999999A, 0x3FF8000000000000], 64)
    _, rows = flow.transfer(narrow, AbstractState(regs={"xmm1": hex(inexact)}),
                            0x1152, collect=True)
    assert rows[0]["write"]["value"] == "?"


def test_simd_lane_extract_and_insert_fold_known_lanes():
    b = block(0x10A0, [
        ins(0x10A0, "mov", "eax, 0x11223344"),
        ins(0x10A1, "movd", "xmm0, eax"),
        ins(0x10A2, "mov", "ebx, 0xabcd"),
        ins(0x10A3, "pinsrw", "xmm0, bx, 1"),
        ins(0x10A4, "pextrw", "ecx, xmm0, 1"),
        ins(0x10A5, "pextrb", "dl, xmm0, 2"),
        ins(0x10A6, "pextrw", "r9w, xmm0, 9"),
        ins(0x10A7, "vpxor", "xmm1, xmm1, xmm1"),
        ins(0x10A8, "vpinsrd", "xmm1, xmm1, eax, 2"),
        ins(0x10A9, "vpextrd", "r8d, xmm1, 2"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x10A0, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x10a3"]["write"]["value"] == "0xabcd3344"
    assert by_address["0x10a4"]["write"]["value"] == "0xabcd"
    assert by_address["0x10a5"]["write"]["value"] == "0xcd"
    assert by_address["0x10a6"]["write"]["value"] == "0xabcd"
    assert by_address["0x10a8"]["write"]["value"] == "0x112233440000000000000000"
    assert by_address["0x10a9"]["write"]["value"] == "0x11223344"


def test_packed_byte_shifts_move_bytes_within_each_128_bit_lane():
    b = block(0x1060, [
        ins(0x1060, "mov", "eax, 0x12345678"),
        ins(0x1061, "movd", "xmm0, eax"),
        ins(0x1062, "pslldq", "xmm0, 1"),
        ins(0x1063, "psrldq", "xmm0, 1"),
        ins(0x1064, "pslldq", "xmm0, 16"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1060, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x1062"]["write"]["value"] == "0x1234567800"
    assert by_address["0x1063"]["write"]["value"] == "0x12345678"
    assert by_address["0x1064"]["write"]["value"] == "0"


def test_setcc_and_cmovcc_use_known_arithmetic_flags():
    b = block(0x1070, [
        ins(0x1070, "mov", "eax, 5"),
        ins(0x1071, "cmp", "eax, 5"),
        ins(0x1072, "sete", "bl"),
        ins(0x1073, "movzx", "ecx, bl"),
        ins(0x1074, "mov", "esi, 0x1234"),
        ins(0x1075, "mov", "edx, 9"),
        ins(0x1076, "cmp", "eax, 4"),
        ins(0x1077, "cmovg", "edx, esi"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1070, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x1072"]["write"]["value"] == "1"
    assert by_address["0x1073"]["write"]["value"] == "1"
    assert by_address["0x1077"]["write"]["value"] == "0x1234"


def test_vex_vector_writes_zero_and_propagate_upper_aliases():
    b = block(0x1080, [
        ins(0x1080, "mov", "eax, 0xffffffff"),
        ins(0x1081, "movd", "xmm0, eax"),
        ins(0x1082, "vpxor", "xmm0, xmm0, xmm0"),
        ins(0x1083, "vmovdqu", "ymm1, ymm0"),
        ins(0x1084, "movd", "eax, xmm1"),
        ins(0x1085, "vpxor", "ymm2, ymm2, ymm2"),
        ins(0x1086, "vpcmpeqd", "xmm2, xmm2, xmm2"),
        ins(0x1087, "vmovdqu", "ymm3, ymm2"),
        ins(0x1088, "movd", "ecx, xmm3"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1080, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x1082"]["additional_writes"] == [
        {"target": "ymm0", "value": "0"}, {"target": "zmm0", "value": "0"},
    ]
    assert by_address["0x1084"]["write"]["value"] == "0"
    assert by_address["0x1086"]["additional_writes"] == [
        {"target": "ymm2", "value": "0xffffffffffffffffffffffffffffffff"},
        {"target": "zmm2", "value": "0xffffffffffffffffffffffffffffffff"},
    ]
    assert by_address["0x1088"]["write"]["value"] == "0xffffffff"


def test_scalar_integer_folding_wraps_at_operand_width():
    b = block(0x1050, [
        ins(0x1050, "mov", "eax, 0xffffffff"),
        ins(0x1051, "add", "eax, 1"),
        ins(0x1052, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1050, [b])
    add = next(row for row in rows if row["address"] == "0x1051")
    assert add["write"]["value"] == "0"


def test_scalar_shift_folding_masks_architectural_count():
    b = block(0x1060, [
        ins(0x1060, "mov", "eax, 1"),
        ins(0x1061, "shl", "eax, 32"),
        ins(0x1062, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1060, [b])
    shift = next(row for row in rows if row["address"] == "0x1061")
    assert shift["write"]["value"] == "1"


def test_scalar_unary_folding_wraps_at_operand_width():
    b = block(0x1070, [
        ins(0x1070, "mov", "eax, 0xffffffff"),
        ins(0x1071, "inc", "eax"),
        ins(0x1072, "mov", "eax, 0x80000000"),
        ins(0x1073, "neg", "eax"),
        ins(0x1074, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1070, [b])
    inc = next(row for row in rows if row["address"] == "0x1071")
    neg = next(row for row in rows if row["address"] == "0x1073")
    assert inc["write"]["value"] == "0"
    assert neg["write"]["value"] == "0x80000000"


def test_adc_and_sbb_use_proven_carry_from_compare():
    b = block(0x1080, [
        ins(0x1080, "mov", "eax, 0"),
        ins(0x1081, "cmp", "eax, 1"),
        ins(0x1082, "adc", "eax, 2"),
        ins(0x1083, "mov", "edx, 0"),
        ins(0x1084, "cmp", "edx, 1"),
        ins(0x1085, "sbb", "edx, 1"),
        ins(0x1086, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1080, [b])
    adc = next(row for row in rows if row["address"] == "0x1082")
    sbb = next(row for row in rows if row["address"] == "0x1085")
    assert adc["write"]["value"] == "3"
    assert sbb["write"]["value"] == "0xfffffffe"
    assert sbb["carry_flag_after"] is True


def test_unknown_adc_carry_invalidates_destination_and_does_not_fold_stale_value():
    b = block(0x1090, [
        ins(0x1090, "mov", "eax, 5"),
        ins(0x1091, "adc", "eax, 1"),
        ins(0x1092, "add", "eax, 1"),
        ins(0x1093, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1090, [b])
    adc = next(row for row in rows if row["address"] == "0x1091")
    add = next(row for row in rows if row["address"] == "0x1092")
    assert adc["write"]["value"] == "?"
    assert add["write"]["value"] == "?"


def test_inc_preserves_carry_for_following_adc():
    b = block(0x10A0, [
        ins(0x10A0, "mov", "eax, 0"),
        ins(0x10A1, "cmp", "eax, 1"),
        ins(0x10A2, "inc", "eax"),
        ins(0x10A3, "adc", "eax, 0"),
        ins(0x10A4, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x10A0, [b])
    adc = next(row for row in rows if row["address"] == "0x10a3")
    assert adc["write"]["value"] == "2"


def test_unmodeled_carry_writer_invalidates_old_carry_before_adc():
    b = block(0x10B0, [
        ins(0x10B0, "mov", "eax, 0"),
        ins(0x10B1, "cmp", "eax, 1"),
        ins(0x10B2, "bt", "ebx, 0"),
        ins(0x10B3, "adc", "eax, 1"),
        ins(0x10B4, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x10B0, [b])
    adc = next(row for row in rows if row["address"] == "0x10b3")
    assert adc["write"]["value"] == "?"


def test_clc_and_stc_set_carry_for_adc_sbb():
    b = block(0x10C0, [
        ins(0x10C0, "mov", "eax, 10"),
        ins(0x10C1, "stc"),
        ins(0x10C2, "adc", "eax, 1"),
        ins(0x10C3, "clc"),
        ins(0x10C4, "sbb", "eax, 1"),
        ins(0x10C5, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x10C0, [b])
    adc = next(row for row in rows if row["address"] == "0x10c2")
    sbb = next(row for row in rows if row["address"] == "0x10c4")
    assert adc["write"]["value"] == "0xc"
    assert sbb["write"]["value"] == "0xb"


def test_carry_condition_branches_fold_when_cf_is_known():
    b = block(0x10D0, [
        ins(0x10D0, "mov", "eax, 0"),
        ins(0x10D1, "cmp", "eax, 1"),
        ins(0x10D2, "jb", "0x10E0", target=0x10E0, flow="conditional"),
        ins(0x10D3, "clc"),
        ins(0x10D4, "jnc", "0x10E0", target=0x10E0, flow="conditional"),
        ins(0x10D5, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x10D0, [b])
    jb = next(row for row in rows if row["address"] == "0x10d2")
    jnc = next(row for row in rows if row["address"] == "0x10d4")
    assert jb["condition"] == "1"
    assert jnc["condition"] == "1"


def test_overlapping_stack_write_invalidates_the_wider_cached_slot():
    b = block(0x1100, [
        ins(0x1100, "mov", "dword ptr [rbp - 4], 0x11223344"),
        ins(0x1101, "mov", "byte ptr [rbp - 2], 0xAA"),
        ins(0x1102, "mov", "eax, dword ptr [rbp - 4]"),
        ins(0x1103, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1100, [b])
    partial_write = next(row for row in rows if row["address"] == "0x1101")
    load = next(row for row in rows if row["address"] == "0x1102")
    assert partial_write["stack_slots_after"] == {"rbp - 2": "0xaa"}
    assert partial_write["stack_slot_widths_after"] == {"rbp - 2": 1}
    assert load["resolved_operands"][1] == "mem[rbp - 4]"


def test_stack_value_is_not_reused_when_read_width_differs():
    b = block(0x1200, [
        ins(0x1200, "mov", "dword ptr [rbp - 8], 0x1234"),
        ins(0x1201, "mov", "rax, qword ptr [rbp - 8]"),
        ins(0x1202, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1200, [b])
    load = next(row for row in rows if row["address"] == "0x1201")
    assert load["resolved_operands"][1] == "?"
    assert load["registers_after"]["rax"] == "?"


def test_nonoverlapping_stack_writes_keep_independent_values():
    b = block(0x1300, [
        ins(0x1300, "mov", "dword ptr [rbp - 8], 11"),
        ins(0x1301, "mov", "dword ptr [rbp - 4], 22"),
        ins(0x1302, "mov", "eax, dword ptr [rbp - 8]"),
        ins(0x1303, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1300, [b])
    second_store = next(row for row in rows if row["address"] == "0x1301")
    load = next(row for row in rows if row["address"] == "0x1302")
    assert second_store["stack_slots_after"] == {"rbp - 8": "0xb", "rbp - 4": "0x16"}
    assert load["resolved_operands"][1] == "0xb"


def test_stack_aliases_share_one_canonical_slot():
    b = block(0x1800, [
        ins(0x1800, "mov", "dword ptr [ebp - 4], 9"),
        ins(0x1801, "mov", "eax, dword ptr [rbp - 4]"),
        ins(0x1802, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(64, {}, {}, {})
    rows, _ = flow.analyze(0x1800, [b])
    load = next(row for row in rows if row["address"] == "0x1801")
    assert load["resolved_operands"][1] == "9"
    assert "rbp - 4" in load["stack_slots_after"]


def test_uninitialized_positive_stack_slots_are_named_as_stack_arguments():
    b = block(0x1900, [
        ins(0x1900, "mov", "eax, dword ptr [rbp + 0x30]"),
        ins(0x1901, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(64, {}, {}, {})
    rows, _ = flow.analyze(0x1900, [b])
    load = next(row for row in rows if row["address"] == "0x1900")
    assert load["resolved_operands"][1] == "stack_arg_30"


def test_value_flows_across_basic_blocks_into_branch_condition():
    b1 = block(0x2000, [
        ins(0x2000, "mov", "dword ptr [rbp - 4], 36"),
        ins(0x2001, "cmp", "dword ptr [rbp - 4], 0x1e"),
        ins(0x2002, "jle", "0x2010", target=0x2010, flow="conditional"),
    ], [0x2010, 0x2003])
    b2 = block(0x2003, [ins(0x2003, "mov", "eax, 1"), ins(0x2004, "ret", flow="return")])
    b3 = block(0x2010, [ins(0x2010, "mov", "eax, 0"), ins(0x2011, "ret", flow="return")])
    flow = FunctionDataFlow(64, {}, {}, {})
    rows, _ = flow.analyze(0x2000, [b1, b2, b3])
    branch = next(r for r in rows if r["address"] == "0x2002")
    assert branch["condition"] == "0"
    assert not any(r["address"] == "0x2010" for r in rows)
    fallthrough = next(r for r in rows if r["address"] == "0x2003")
    assert fallthrough["write"]["value"] == "1"


def test_proven_conditional_edge_preserves_values_at_join():
    entry = block(0x2100, [
        ins(0x2100, "mov", "eax, 7"),
        ins(0x2101, "cmp", "eax, 7"),
        ins(0x2102, "je", "0x2110", target=0x2110, flow="conditional"),
    ], [0x2110, 0x2103])
    fallthrough = block(0x2103, [
        ins(0x2103, "mov", "ebx, 2"),
        ins(0x2104, "jmp", "0x2120", target=0x2120, flow="jump"),
    ], [0x2120])
    taken = block(0x2110, [
        ins(0x2110, "mov", "ebx, 1"),
        ins(0x2111, "jmp", "0x2120", target=0x2120, flow="jump"),
    ], [0x2120])
    join = block(0x2120, [ins(0x2120, "mov", "ecx, ebx"), ins(0x2121, "ret", flow="return")])
    rows, summary = FunctionDataFlow(64, {}, {}, {}).analyze(0x2100, [entry, fallthrough, taken, join])
    by_address = {row["address"]: row for row in rows}
    assert "0x2103" not in by_address
    assert by_address["0x2120"]["write"]["value"] == "1"
    assert summary["blocks_reachable"] == 3


def test_loop_family_updates_count_and_prunes_proven_edges():
    loop_entry = block(0x2200, [
        ins(0x2200, "mov", "ecx, 1"),
        ins(0x2201, "loop", "0x2210", target=0x2210, flow="conditional"),
    ], [0x2210, 0x2202])
    loop_fallthrough = block(0x2202, [ins(0x2202, "ret", flow="return")])
    loop_target = block(0x2210, [ins(0x2210, "ret", flow="return")])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(
        0x2200, [loop_entry, loop_fallthrough, loop_target]
    )
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x2201"]["condition"] == "0"
    assert by_address["0x2201"]["write"]["value"] == "0"
    assert "0x2210" not in by_address

    count_entry = block(0x2220, [
        ins(0x2220, "mov", "ecx, 0"),
        ins(0x2221, "jecxz", "0x2230", target=0x2230, flow="conditional"),
    ], [0x2230, 0x2222])
    count_fallthrough = block(0x2222, [ins(0x2222, "ret", flow="return")])
    count_target = block(0x2230, [ins(0x2230, "ret", flow="return")])
    count_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(
        0x2220, [count_entry, count_fallthrough, count_target]
    )
    assert not any(row["address"] == "0x2222" for row in count_rows)

    repe_entry = block(0x2240, [
        ins(0x2240, "mov", "ecx, 2"),
        ins(0x2241, "mov", "eax, 5"),
        ins(0x2242, "cmp", "eax, 5"),
        ins(0x2243, "loope", "0x2250", target=0x2250, flow="conditional"),
    ], [0x2250, 0x2244])
    repe_fallthrough = block(0x2244, [ins(0x2244, "ret", flow="return")])
    repe_target = block(0x2250, [ins(0x2250, "ret", flow="return")])
    repe_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(
        0x2240, [repe_entry, repe_fallthrough, repe_target]
    )
    repe_by_address = {row["address"]: row for row in repe_rows}
    assert repe_by_address["0x2243"]["condition"] == "1"
    assert "0x2244" not in repe_by_address


def test_x87_stack_load_arithmetic_store_and_unknown_widening():
    b = block(0x2300, [
        ins(0x2300, "fld1"),
        ins(0x2301, "fld1"),
        ins(0x2302, "faddp", "st(1), st(0)"),
        ins(0x2303, "fstp", "qword ptr [rbp - 8]"),
        ins(0x2304, "fld", "qword ptr [rbp - 8]"),
        ins(0x2305, "fscale"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x2300, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x2300"]["x87_stack_after"] == ["1.0"]
    assert by_address["0x2302"]["x87_stack_after"] == ["2.0"]
    assert by_address["0x2303"]["write"]["value"] == "2.0"
    assert by_address["0x2303"]["x87_stack_after"] == []
    assert by_address["0x2304"]["x87_stack_after"] == ["2.0"]
    assert by_address["0x2305"]["x87_stack_after"] is None


def test_x87_integer_conversions_and_fcmov_flags():
    b = block(0x2310, [
        ins(0x2310, "mov", "dword ptr [rbp - 4], 7"),
        ins(0x2311, "fild", "dword ptr [rbp - 4]"),
        ins(0x2312, "fld1"),
        ins(0x2313, "fcomi", "st(0), st(1)"),
        ins(0x2314, "fcmovb", "st(0), st(1)"),
        ins(0x2315, "fistp", "dword ptr [rbp - 8]"),
        ins(0x2316, "fild", "dword ptr [rbp - 4]"),
        ins(0x2317, "fld1"),
        ins(0x2318, "fcomip", "st(0), st(1)"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x2310, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x2311"]["x87_stack_after"] == ["7.0"]
    assert by_address["0x2313"]["carry_flag_after"] is True
    assert by_address["0x2314"]["x87_stack_after"] == ["7.0", "7.0"]
    assert by_address["0x2315"]["write"]["value"] == "round_to_int_using_x87_control_word(7.0)"
    assert by_address["0x2315"]["x87_stack_after"] == ["7.0"]
    assert by_address["0x2318"]["carry_flag_after"] is True
    assert by_address["0x2318"]["x87_stack_after"] == ["7.0", "7.0"]


def test_x87_exact_integer_arithmetic_folds_and_respects_reverse_subtract():
    b = block(0x2320, [
        ins(0x2320, "fld1"),
        ins(0x2321, "fld1"),
        ins(0x2322, "faddp", "st(1), st(0)"),
        ins(0x2323, "fld1"),
        ins(0x2324, "fsubrp", "st(1), st(0)"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x2320, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x2322"]["x87_stack_after"] == ["2.0"]
    assert by_address["0x2324"]["x87_stack_after"] == ["-1.0"]


def test_call_result_uses_direct_target_and_inferred_arity():
    b = block(0x3000, [
        ins(0x3000, "mov", "ecx, 6"),
        ins(0x3001, "call", "0x4000", target=0x4000, flow="call"),
        ins(0x3002, "mov", "dword ptr [rbp - 4], eax"),
        ins(0x3003, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(64, {}, {}, {0x4000: "square"}, function_arg_counts={0x4000: 1})
    rows, _ = flow.analyze(0x3000, [b])
    store = next(r for r in rows if r["address"] == "0x3002")
    assert store["write"]["value"] == "square(6)"
    assert store["stack_slots_after"]["rbp - 4"] == "square(6)"


def test_call_result_type_uses_recovered_callee_signature():
    b = block(0x3200, [
        ins(0x3200, "call", "0x4000", target=0x4000, flow="call"),
        ins(0x3201, "mov", "dword ptr [rbp - 4], eax"),
        ins(0x3202, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(
        64, {}, {}, {0x4000: "get_count"},
        function_return_types={0x4000: "uint32_t"},
    )
    rows, _ = flow.analyze(0x3200, [b])
    call = next(row for row in rows if row["address"] == "0x3200")
    store = next(row for row in rows if row["address"] == "0x3201")
    assert call["write_type"] == "uint32_t"
    assert store["write_type"] == "uint32_t"


def test_known_allocation_and_buffer_api_returns_recover_pointer_types():
    b = block(0x3210, [
        ins(0x3210, "call", "0x4000", target=0x4000, flow="call"),
        ins(0x3211, "mov", "qword ptr [rbp - 8], rax"),
        ins(0x3212, "call", "0x4010", target=0x4010, flow="call"),
        ins(0x3213, "call", "0x4020", target=0x4020, flow="call"),
        ins(0x3214, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(
        64, {}, {}, {0x4000: "calloc", 0x4010: "memcpy", 0x4020: "free"}
    ).analyze(0x3210, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x3210"]["write_type"] == "uintptr_t"
    assert by_address["0x3211"]["write_type"] == "uintptr_t"
    assert by_address["0x3212"]["write_type"] == "uintptr_t"
    assert by_address["0x3213"]["write_type"] == "void"


def test_indirect_tail_jump_resolves_only_to_known_code_addresses():
    b = block(0x3230, [
        ins(0x3230, "mov", "rax, 0x4000"),
        ins(0x3231, "jmp", "rax", flow="jump"),
    ])
    rows, _ = FunctionDataFlow(
        64, {}, {}, {}, known_code_addresses={0x4000}
    ).analyze(0x3230, [b])
    jump = next(row for row in rows if row["address"] == "0x3231")
    assert jump["resolved_indirect_jump_target"] == "0x4000"

    table = block(0x3240, [
        ins(0x3240, "jmp", "qword ptr [rax + rcx*8]", flow="jump"),
    ])
    rows, _ = FunctionDataFlow(
        64, {}, {}, {}, known_code_addresses={0x4000}
    ).analyze(0x3240, [table])
    jump = next(row for row in rows if row["address"] == "0x3240")
    assert jump["resolved_indirect_jump_target"] is None


def test_pointer_type_propagates_through_nonstack_memory_loads():
    b = block(0x3220, [
        ins(0x3220, "call", "0x4000", target=0x4000, flow="call"),
        ins(0x3221, "mov", "rbx, rax"),
        ins(0x3222, "mov", "qword ptr [rbx], rax"),
        ins(0x3223, "mov", "rcx, qword ptr [rbx]"),
        ins(0x3224, "mov", "qword ptr [rbp - 8], rcx"),
        ins(0x3225, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {0x4000: "malloc"}).analyze(0x3220, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x3222"]["memory_types_after"]["rbx"] == "uintptr_t"
    assert by_address["0x3223"]["write_type"] == "uintptr_t"
    assert by_address["0x3224"]["write_type"] == "uintptr_t"


def test_common_win32_handle_import_has_pointer_return_type():
    call = Instruction(
        0x3238, 1, "90", "call", "rax", ".text", flow="call",
        import_name="KERNEL32.dll!GetModuleHandleW",
    )
    flow = FunctionDataFlow(64, {}, {}, {})
    rows, _ = flow.analyze(0x3238, [block(0x3238, [call])])
    assert rows[0]["write_type"] == "uintptr_t"
    assert rows[0]["call_argument_types"] == ["const wchar_t *"]


def test_merge_drops_values_that_disagree_between_paths():
    left = AbstractState(regs={"rax": "5", "rcx": "arg1"}, stack={"rbp - 4": "6"})
    right = AbstractState(regs={"rax": "7", "rcx": "arg1"}, stack={"rbp - 4": "6"})
    merged = AbstractState.merge(left, right)
    assert "rax" not in merged.regs
    assert merged.regs["rcx"] == "arg1"
    assert merged.stack["rbp - 4"] == "6"


def test_merge_drops_stack_facts_with_conflicting_access_widths():
    left = AbstractState(stack={"rbp - 4": "same"}, stack_widths={"rbp - 4": 4})
    right = AbstractState(stack={"rbp - 4": "same"}, stack_widths={"rbp - 4": 8})
    merged = AbstractState.merge(left, right)
    assert "rbp - 4" not in merged.stack
    assert "rbp - 4" not in merged.stack_widths


def test_merge_widens_conflicting_types_when_value_agrees():
    left = AbstractState(regs={"rax": "same"}, reg_types={"rax": "uint32_t"})
    right = AbstractState(regs={"rax": "same"}, reg_types={"rax": "uintptr_t"})
    merged = AbstractState.merge(left, right)
    assert merged.regs["rax"] == "same"
    assert merged.reg_types["rax"] == "unknown"


def test_merge_widens_conflicting_nonstack_memory_types():
    key = "symbol:arg1+8"
    left = AbstractState(
        memory={key: "same"}, memory_widths={key: 8},
        memory_types={key: "uintptr_t"},
    )
    right = AbstractState(
        memory={key: "same"}, memory_widths={key: 8},
        memory_types={key: "uint64_t"},
    )
    merged = AbstractState.merge(left, right)
    assert merged.memory[key] == "same"
    assert merged.memory_types[key] == "unknown"


def test_merge_discards_disagreeing_carry_facts():
    left = AbstractState(carry=True)
    right = AbstractState(carry=False)
    assert AbstractState.merge(left, right).carry is None


def test_add_overflow_sign_zero_and_signed_branch_flags_are_tracked():
    b = block(0x10E0, [
        ins(0x10E0, "mov", "eax, 0x7fffffff"),
        ins(0x10E1, "add", "eax, 1"),
        ins(0x10E2, "jo", "0x10F0", target=0x10F0, flow="conditional"),
        ins(0x10E3, "js", "0x10F0", target=0x10F0, flow="conditional"),
        ins(0x10E4, "jg", "0x10F0", target=0x10F0, flow="conditional"),
        ins(0x10E5, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x10E0, [b])
    add = next(row for row in rows if row["address"] == "0x10e1")
    jo = next(row for row in rows if row["address"] == "0x10e2")
    js = next(row for row in rows if row["address"] == "0x10e3")
    jg = next(row for row in rows if row["address"] == "0x10e4")
    assert add["write"]["value"] == "0x80000000"
    assert add["overflow_flag_after"] is True
    assert add["sign_flag_after"] is True
    assert add["zero_flag_after"] is False
    assert jo["condition"] == "1"
    assert js["condition"] == "1"
    assert jg["condition"] == "1"


def test_logic_test_sets_zero_parity_and_clears_overflow():
    b = block(0x1100, [
        ins(0x1100, "mov", "eax, 1"),
        ins(0x1101, "test", "al, 1"),
        ins(0x1102, "jnp", "0x1110", target=0x1110, flow="conditional"),
        ins(0x1103, "je", "0x1110", target=0x1110, flow="conditional"),
        ins(0x1104, "jo", "0x1110", target=0x1110, flow="conditional"),
        ins(0x1105, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1100, [b])
    test_row = next(row for row in rows if row["address"] == "0x1101")
    jnp = next(row for row in rows if row["address"] == "0x1102")
    je = next(row for row in rows if row["address"] == "0x1103")
    jo = next(row for row in rows if row["address"] == "0x1104")
    assert test_row["zero_flag_after"] is False
    assert test_row["parity_flag_after"] is False
    assert test_row["overflow_flag_after"] is False
    assert jnp["condition"] == "1"
    assert je["condition"] == "0"
    assert jo["condition"] == "0"


def test_single_bit_shift_tracks_carry_overflow_and_result_flags():
    b = block(0x1120, [
        ins(0x1120, "mov", "al, 0x80"),
        ins(0x1121, "shl", "al, 1"),
        ins(0x1122, "jc", "0x1130", target=0x1130, flow="conditional"),
        ins(0x1123, "jo", "0x1130", target=0x1130, flow="conditional"),
        ins(0x1124, "je", "0x1130", target=0x1130, flow="conditional"),
        ins(0x1125, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1120, [b])
    shift = next(row for row in rows if row["address"] == "0x1121")
    jc = next(row for row in rows if row["address"] == "0x1122")
    jo = next(row for row in rows if row["address"] == "0x1123")
    je = next(row for row in rows if row["address"] == "0x1124")
    assert shift["write"]["value"] == "0"
    assert shift["carry_flag_after"] is True
    assert shift["overflow_flag_after"] is True
    assert shift["zero_flag_after"] is True
    assert jc["condition"] == "1"
    assert jo["condition"] == "1"
    assert je["condition"] == "1"


def test_zero_effective_shift_count_preserves_prior_flags():
    b = block(0x1140, [
        ins(0x1140, "mov", "eax, 0"),
        ins(0x1141, "cmp", "eax, 1"),
        ins(0x1142, "sar", "eax, 32"),
        ins(0x1143, "jc", "0x1150", target=0x1150, flow="conditional"),
        ins(0x1144, "je", "0x1150", target=0x1150, flow="conditional"),
        ins(0x1145, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1140, [b])
    shift = next(row for row in rows if row["address"] == "0x1142")
    jc = next(row for row in rows if row["address"] == "0x1143")
    je = next(row for row in rows if row["address"] == "0x1144")
    assert shift["write"]["value"] == "0"
    assert shift["carry_flag_after"] is True
    assert jc["condition"] == "1"
    assert je["condition"] == "0"


def test_shift_count_beyond_operand_width_clears_undefined_carry_and_overflow():
    b = block(0x1160, [
        ins(0x1160, "mov", "al, 1"),
        ins(0x1161, "shl", "al, 9"),
        ins(0x1162, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1160, [b])
    shift = next(row for row in rows if row["address"] == "0x1161")
    assert shift["write"]["value"] == "0"
    assert shift["carry_flag_after"] is None
    assert shift["overflow_flag_after"] is None
    assert shift["zero_flag_after"] is True


def test_unsigned_single_operand_mul_writes_accumulator_and_high_register():
    b = block(0x1180, [
        ins(0x1180, "mov", "eax, 0xffffffff"),
        ins(0x1181, "mov", "ecx, 2"),
        ins(0x1182, "mul", "ecx"),
        ins(0x1183, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1180, [b])
    mul = next(row for row in rows if row["address"] == "0x1182")
    assert mul["registers_after"]["rax"] == "0xfffffffe"
    assert mul["registers_after"]["rdx"] == "1"
    assert mul["carry_flag_after"] is True
    assert mul["overflow_flag_after"] is True
    assert mul["additional_writes"] == [{"target": "rdx", "value": "1"}]


def test_signed_single_operand_imul_writes_signed_high_half_and_overflow_flags():
    b = block(0x1190, [
        ins(0x1190, "mov", "eax, 0xfffffffe"),
        ins(0x1191, "mov", "ecx, 3"),
        ins(0x1192, "imul", "ecx"),
        ins(0x1193, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1190, [b])
    imul = next(row for row in rows if row["address"] == "0x1192")
    assert imul["registers_after"]["rax"] == "0xfffffffa"
    assert imul["registers_after"]["rdx"] == "0xffffffff"
    assert imul["overflow_flag_after"] is False


def test_two_operand_imul_tracks_signed_overflow():
    b = block(0x11A0, [
        ins(0x11A0, "mov", "eax, 0x40000000"),
        ins(0x11A1, "imul", "eax, 2"),
        ins(0x11A2, "jo", "0x11B0", target=0x11B0, flow="conditional"),
        ins(0x11A3, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x11A0, [b])
    imul = next(row for row in rows if row["address"] == "0x11a1")
    jo = next(row for row in rows if row["address"] == "0x11a2")
    assert imul["registers_after"]["rax"] == "0x80000000"
    assert imul["overflow_flag_after"] is True
    assert jo["condition"] == "1"


def test_unsigned_divide_and_signed_idivide_write_quotient_and_remainder():
    unsigned = block(0x11C0, [
        ins(0x11C0, "mov", "eax, 100"),
        ins(0x11C1, "mov", "edx, 0"),
        ins(0x11C2, "mov", "ecx, 7"),
        ins(0x11C3, "div", "ecx"),
        ins(0x11C4, "ret", flow="return"),
    ])
    signed = block(0x11D0, [
        ins(0x11D0, "mov", "eax, 0xfffffff9"),
        ins(0x11D1, "mov", "edx, 0xffffffff"),
        ins(0x11D2, "mov", "ecx, 3"),
        ins(0x11D3, "idiv", "ecx"),
        ins(0x11D4, "ret", flow="return"),
    ])
    urows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x11C0, [unsigned])
    srows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x11D0, [signed])
    udiv = next(row for row in urows if row["address"] == "0x11c3")
    idiv = next(row for row in srows if row["address"] == "0x11d3")
    assert udiv["registers_after"]["rax"] == "0xe"
    assert udiv["registers_after"]["rdx"] == "2"
    assert idiv["registers_after"]["rax"] == "0xfffffffe"
    assert idiv["registers_after"]["rdx"] == "0xffffffff"
    assert idiv["carry_flag_after"] is None


def test_divide_by_zero_invalidates_implicit_results():
    b = block(0x11E0, [
        ins(0x11E0, "mov", "eax, 100"),
        ins(0x11E1, "mov", "edx, 0"),
        ins(0x11E2, "mov", "ecx, 0"),
        ins(0x11E3, "div", "ecx"),
        ins(0x11E4, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x11E0, [b])
    div = next(row for row in rows if row["address"] == "0x11e3")
    assert div["registers_after"]["rax"] == "?"
    assert div["registers_after"]["rdx"] == "?"


def test_eight_bit_mul_and_div_use_ax_as_implicit_double_width_register():
    multiply = block(0x11F0, [
        ins(0x11F0, "mov", "eax, 0x103"),
        ins(0x11F1, "mov", "ecx, 2"),
        ins(0x11F2, "mul", "cl"),
        ins(0x11F3, "ret", flow="return"),
    ])
    divide = block(0x1200, [
        ins(0x1200, "mov", "eax, 0x123"),
        ins(0x1201, "mov", "ecx, 0x10"),
        ins(0x1202, "div", "cl"),
        ins(0x1203, "ret", flow="return"),
    ])
    mrows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x11F0, [multiply])
    drows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1200, [divide])
    mul = next(row for row in mrows if row["address"] == "0x11f2")
    div = next(row for row in drows if row["address"] == "0x1202")
    assert mul["registers_after"]["rax"] == "6"
    assert div["registers_after"]["rax"] == "0x312"


def test_unsigned_divide_quotient_overflow_invalidates_both_results():
    b = block(0x1210, [
        ins(0x1210, "mov", "eax, 0x100"),
        ins(0x1211, "mov", "edx, 0"),
        ins(0x1212, "mov", "ecx, 1"),
        ins(0x1213, "div", "cl"),
        ins(0x1214, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1210, [b])
    div = next(row for row in rows if row["address"] == "0x1213")
    assert div["registers_after"]["rax"] == "?"
    # Byte division writes AX only; the RDX family is untouched.
    assert div["registers_after"]["rdx"] == "0"


def test_register_xchg_swaps_known_values_and_memory_xadd_invalidates_stack_fact():
    swap = block(0x1220, [
        ins(0x1220, "mov", "eax, 5"),
        ins(0x1221, "mov", "ecx, 9"),
        ins(0x1222, "xchg", "eax, ecx"),
        ins(0x1223, "ret", flow="return"),
    ])
    atomic = block(0x1230, [
        ins(0x1230, "mov", "dword ptr [rbp - 4], 7"),
        ins(0x1231, "xadd", "dword ptr [rbp - 4], eax"),
        ins(0x1232, "mov", "ecx, dword ptr [rbp - 4]"),
        ins(0x1233, "ret", flow="return"),
    ])
    swap_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1220, [swap])
    atomic_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1230, [atomic])
    swap_row = next(row for row in swap_rows if row["address"] == "0x1222")
    atomic_row = next(row for row in atomic_rows if row["address"] == "0x1231")
    read_row = next(row for row in atomic_rows if row["address"] == "0x1232")
    assert swap_row["registers_after"]["rax"] == "9"
    assert swap_row["registers_after"]["rcx"] == "5"
    assert atomic_row["registers_after"]["rax"] == "?"
    assert read_row["resolved_operands"][1] != "7"


def test_register_bit_test_updates_carry_and_bit_modify_result():
    b = block(0x1240, [
        ins(0x1240, "mov", "eax, 5"),
        ins(0x1241, "bt", "eax, 2"),
        ins(0x1242, "jc", "0x1244", target=0x1244, flow="conditional"),
        ins(0x1243, "bts", "eax, 1"),
        ins(0x1244, "btr", "eax, 0"),
        ins(0x1245, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1240, [b])
    bt = next(row for row in rows if row["address"] == "0x1241")
    branch = next(row for row in rows if row["address"] == "0x1242")
    bts = next(row for row in rows if row["address"] == "0x1243")
    btr = next(row for row in rows if row["address"] == "0x1244")
    assert bt["carry_flag_after"] is True
    assert branch["condition"] == "1"
    assert bts["registers_after"]["rax"] == "7"
    assert btr["registers_after"]["rax"] == "6"
    assert bts["carry_flag_after"] is False
    assert btr["carry_flag_after"] is True


def test_not_and_bswap_results_preserve_arithmetic_flags():
    b = block(0x1250, [
        ins(0x1250, "mov", "eax, 0x0f"),
        ins(0x1251, "cmp", "eax, 0x0f"),
        ins(0x1252, "not", "eax"),
        ins(0x1253, "je", "0x1256", target=0x1256, flow="conditional"),
        ins(0x1254, "mov", "ebx, 0x12345678"),
        ins(0x1255, "bswap", "ebx"),
        ins(0x1256, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1250, [b])
    not_row = next(row for row in rows if row["address"] == "0x1252")
    branch = next(row for row in rows if row["address"] == "0x1253")
    bswap = next(row for row in rows if row["address"] == "0x1255")
    assert not_row["write"]["value"] == "0xfffffff0"
    assert branch["condition"] == "1"
    assert bswap["write"]["value"] == "0x78563412"


def test_rotate_flags_and_zero_effective_count():
    b = block(0x1260, [
        ins(0x1260, "mov", "eax, 0x81"),
        ins(0x1261, "cmp", "eax, 0x81"),
        ins(0x1262, "rol", "al, 1"),
        ins(0x1263, "jc", "0x1268", target=0x1268, flow="conditional"),
        ins(0x1264, "ror", "al, 1"),
        ins(0x1265, "rol", "al, 0"),
        ins(0x1266, "jz", "0x1268", target=0x1268, flow="conditional"),
        ins(0x1267, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1260, [b])
    rol = next(row for row in rows if row["address"] == "0x1262")
    branch_carry = next(row for row in rows if row["address"] == "0x1263")
    ror = next(row for row in rows if row["address"] == "0x1264")
    zero_count = next(row for row in rows if row["address"] == "0x1265")
    branch_zero = next(row for row in rows if row["address"] == "0x1266")
    assert rol["write"]["value"] == "3"
    assert rol["carry_flag_after"] is True
    assert rol["overflow_flag_after"] is True
    assert branch_carry["condition"] == "1"
    assert ror["write"]["value"] == "0x81"
    assert zero_count["carry_flag_after"] is True
    assert zero_count["zero_flag_after"] is True
    assert branch_zero["condition"] == "1"


def test_carry_rotates_use_and_update_the_incoming_carry():
    left = block(0x1270, [
        ins(0x1270, "mov", "al, 0x80"),
        ins(0x1271, "stc"),
        ins(0x1272, "rcl", "al, 1"),
        ins(0x1273, "ret", flow="return"),
    ])
    right = block(0x1280, [
        ins(0x1280, "mov", "al, 1"),
        ins(0x1281, "clc"),
        ins(0x1282, "rcr", "al, 1"),
        ins(0x1283, "ret", flow="return"),
    ])
    left_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1270, [left])
    right_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1280, [right])
    rcl = next(row for row in left_rows if row["address"] == "0x1272")
    rcr = next(row for row in right_rows if row["address"] == "0x1282")
    assert rcl["write"]["value"] == "1"
    assert rcl["carry_flag_after"] is True
    assert rcl["overflow_flag_after"] is True
    assert rcr["write"]["value"] == "0"
    assert rcr["carry_flag_after"] is True


def test_rep_string_effects_widen_implicit_indices_and_memory():
    move = block(0x1290, [
        ins(0x1290, "mov", "esi, 0x1000"),
        ins(0x1291, "mov", "edi, 0x2000"),
        ins(0x1292, "mov", "ecx, 3"),
        ins(0x1293, "mov", "dword ptr [rbp - 4], 7"),
        ins(0x1294, "rep movsb"),
        ins(0x1295, "ret", flow="return"),
    ])
    compare = block(0x12A0, [
        ins(0x12A0, "mov", "ecx, 3"),
        ins(0x12A1, "repe cmpsb"),
        ins(0x12A2, "ret", flow="return"),
    ])
    move_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1290, [move])
    compare_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x12A0, [compare])
    rep = next(row for row in move_rows if row["address"] == "0x1294")
    repe = next(row for row in compare_rows if row["address"] == "0x12a1")
    assert rep["registers_after"]["rsi"] == "?"
    assert rep["registers_after"]["rdi"] == "?"
    assert rep["registers_after"]["rcx"] == "0"
    assert rep["stack_slots_after"] == {}
    assert repe["registers_after"]["rcx"] == "?"
    assert repe["zero_flag_after"] is None


def test_rep_movsb_copies_bounded_known_bytes_and_preserves_disjoint_facts():
    b = block(0x12A8, [
        ins(0x12A8, "mov", "byte ptr [0x1000], 0x41"),
        ins(0x12A9, "mov", "byte ptr [0x1001], 0x42"),
        ins(0x12AA, "mov", "byte ptr [0x3000], 0x77"),
        ins(0x12AB, "mov", "dword ptr [rbp - 4], 7"),
        ins(0x12AC, "mov", "esi, 0x1000"),
        ins(0x12AD, "mov", "edi, 0x2000"),
        ins(0x12AE, "mov", "ecx, 2"),
        ins(0x12AF, "cld"),
        ins(0x12B0, "rep movsb"),
        ins(0x12B1, "mov", "al, byte ptr [0x2000]"),
        ins(0x12B2, "mov", "bl, byte ptr [0x2001]"),
        ins(0x12B3, "mov", "cl, byte ptr [0x3000]"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x12A8, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x12b1"]["write"]["value"] == "0x41"
    assert by_address["0x12b2"]["write"]["value"] == "0x42"
    assert by_address["0x12b3"]["write"]["value"] == "0x77"
    assert by_address["0x12b0"]["stack_slots_after"] == {"rbp - 4": "7"}


def test_rep_stos_fills_values_and_rep_ins_invalidates_only_bounded_range():
    b = block(0x12B8, [
        ins(0x12B8, "mov", "byte ptr [0x4000], 0x55"),
        ins(0x12B9, "mov", "al, 0xaa"),
        ins(0x12BA, "mov", "edi, 0x2000"),
        ins(0x12BB, "mov", "ecx, 3"),
        ins(0x12BC, "cld"),
        ins(0x12BD, "rep stosb"),
        ins(0x12BE, "mov", "dl, byte ptr [0x2002]"),
        ins(0x12BF, "mov", "edi, 0x3000"),
        ins(0x12C0, "mov", "ecx, 2"),
        ins(0x12C1, "rep insb"),
        ins(0x12C2, "mov", "bl, byte ptr [0x3000]"),
        ins(0x12C3, "mov", "cl, byte ptr [0x4000]"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x12B8, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x12bd"]["memory_after"]["va:0x2002"] == "0xaa"
    assert by_address["0x12be"]["write"]["value"] == "0xaa"
    assert by_address["0x12c2"]["write"]["value"] == "?"
    assert by_address["0x12c3"]["write"]["value"] == "0x55"


def test_repe_cmps_and_repne_scas_fold_flags_and_stop_positions():
    b = block(0x12D0, [
        ins(0x12D0, "mov", "byte ptr [0x1000], 1"),
        ins(0x12D1, "mov", "byte ptr [0x1001], 2"),
        ins(0x12D2, "mov", "byte ptr [0x2000], 1"),
        ins(0x12D3, "mov", "byte ptr [0x2001], 9"),
        ins(0x12D4, "mov", "byte ptr [0x3000], 1"),
        ins(0x12D5, "mov", "byte ptr [0x3001], 9"),
        ins(0x12D6, "mov", "esi, 0x1000"),
        ins(0x12D7, "mov", "edi, 0x2000"),
        ins(0x12D8, "mov", "ecx, 3"),
        ins(0x12D9, "cld"),
        ins(0x12DA, "repe cmpsb"),
        ins(0x12DB, "jne", "0x1300", target=0x1300, flow="conditional"),
        ins(0x12DC, "mov", "al, 9"),
        ins(0x12DD, "mov", "edi, 0x3000"),
        ins(0x12DE, "mov", "ecx, 3"),
        ins(0x12DF, "repne scasb"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x12D0, [b])
    by_address = {row["address"]: row for row in rows}
    compare = by_address["0x12da"]
    scan = by_address["0x12df"]
    assert compare["registers_after"]["rsi"] == "0x1002"
    assert compare["registers_after"]["rdi"] == "0x2002"
    assert compare["registers_after"]["rcx"] == "1"
    assert compare["carry_flag_after"] is True and compare["zero_flag_after"] is False
    assert by_address["0x12db"]["condition"] == "1"
    assert scan["registers_after"]["rdi"] == "0x3002"
    assert scan["registers_after"]["rcx"] == "1"
    assert scan["carry_flag_after"] is False and scan["zero_flag_after"] is True


def test_register_xadd_and_cmpxchg_compute_known_results_and_flags():
    xadd_block = block(0x12B0, [
        ins(0x12B0, "mov", "eax, 0xffffffff"),
        ins(0x12B1, "mov", "ebx, 1"),
        ins(0x12B2, "xadd", "eax, ebx"),
        ins(0x12B3, "jc", "0x12B5", target=0x12B5, flow="conditional"),
        ins(0x12B4, "ret", flow="return"),
    ])
    success = block(0x12C0, [
        ins(0x12C0, "mov", "eax, 5"),
        ins(0x12C1, "mov", "ebx, 9"),
        ins(0x12C2, "mov", "ecx, 5"),
        ins(0x12C3, "cmpxchg", "ecx, ebx"),
        ins(0x12C4, "ret", flow="return"),
    ])
    failure = block(0x12D0, [
        ins(0x12D0, "mov", "eax, 5"),
        ins(0x12D1, "mov", "ebx, 9"),
        ins(0x12D2, "mov", "ecx, 4"),
        ins(0x12D3, "cmpxchg", "ecx, ebx"),
        ins(0x12D4, "ret", flow="return"),
    ])
    xadd_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x12B0, [xadd_block])
    success_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x12C0, [success])
    failure_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x12D0, [failure])
    xadd = next(row for row in xadd_rows if row["address"] == "0x12b2")
    branch = next(row for row in xadd_rows if row["address"] == "0x12b3")
    cmp_success = next(row for row in success_rows if row["address"] == "0x12c3")
    cmp_failure = next(row for row in failure_rows if row["address"] == "0x12d3")
    assert xadd["registers_after"]["rax"] == "0"
    assert xadd["registers_after"]["rbx"] == "0xffffffff"
    assert branch["condition"] == "1"
    assert cmp_success["registers_after"]["rcx"] == "9"
    assert cmp_success["registers_after"]["rax"] == "5"
    assert cmp_success["zero_flag_after"] is True
    assert cmp_failure["registers_after"]["rcx"] == "4"
    assert cmp_failure["registers_after"]["rax"] == "4"
    assert cmp_failure["zero_flag_after"] is False


def test_neg_computes_width_limited_result_and_defined_arithmetic_flags():
    zero = block(0x12A0, [
        ins(0x12A0, "mov", "eax, 0"),
        ins(0x12A1, "neg", "eax"),
        ins(0x12A2, "je", "0x12A4", target=0x12A4, flow="conditional"),
        ins(0x12A3, "mov", "ecx, 1"),
        ins(0x12A4, "ret", flow="return"),
    ])
    minimum = block(0x12A8, [
        ins(0x12A8, "mov", "eax, 0x80000000"),
        ins(0x12A9, "neg", "eax"),
    ])
    zero_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x12A0, [zero])
    minimum_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x12A8, [minimum])
    zero_neg = next(row for row in zero_rows if row["address"] == "0x12a1")
    zero_branch = next(row for row in zero_rows if row["address"] == "0x12a2")
    minimum_neg = next(row for row in minimum_rows if row["address"] == "0x12a9")
    assert zero_neg["registers_after"]["rax"] == "0"
    assert zero_neg["carry_flag_after"] is False
    assert zero_neg["zero_flag_after"] is True
    assert zero_neg["overflow_flag_after"] is False
    assert zero_branch["condition"] == "1"
    assert minimum_neg["registers_after"]["rax"] == "0x80000000"
    assert minimum_neg["carry_flag_after"] is True
    assert minimum_neg["overflow_flag_after"] is True


def test_shld_shrd_compute_double_precision_results_and_count_sensitive_flags():
    left = block(0x12B8, [
        ins(0x12B8, "mov", "eax, 0x12345678"),
        ins(0x12B9, "mov", "ebx, 0xabcdef01"),
        ins(0x12BA, "shld", "eax, ebx, 4"),
    ])
    right = block(0x12C8, [
        ins(0x12C8, "mov", "eax, 0x12345678"),
        ins(0x12C9, "mov", "ebx, 0xabcdef01"),
        ins(0x12CA, "shrd", "eax, ebx, 4"),
    ])
    one_bit = block(0x12D8, [
        ins(0x12D8, "mov", "eax, 0x80000000"),
        ins(0x12D9, "mov", "ebx, 0"),
        ins(0x12DA, "shld", "eax, ebx, 1"),
    ])
    zero_count = block(0x12E8, [
        ins(0x12E8, "stc"),
        ins(0x12E9, "mov", "eax, 0x1234"),
        ins(0x12EA, "mov", "ebx, 0"),
        ins(0x12EB, "shld", "eax, ebx, 0"),
    ])
    oversized = block(0x12F8, [
        ins(0x12F8, "mov", "ax, 0x1234"),
        ins(0x12F9, "mov", "bx, 0"),
        ins(0x12FA, "shrd", "ax, bx, 17"),
    ])
    results = [
        FunctionDataFlow(64, {}, {}, {}).analyze(b.start, [b])[0]
        for b in (left, right, one_bit, zero_count, oversized)
    ]
    rows = [{row["address"]: row for row in group} for group in results]
    assert rows[0]["0x12ba"]["registers_after"]["rax"] == "0x2345678a"
    assert rows[0]["0x12ba"]["carry_flag_after"] is True
    assert rows[0]["0x12ba"]["overflow_flag_after"] is None
    assert rows[1]["0x12ca"]["registers_after"]["rax"] == "0x11234567"
    assert rows[1]["0x12ca"]["carry_flag_after"] is True
    assert rows[2]["0x12da"]["registers_after"]["rax"] == "0"
    assert rows[2]["0x12da"]["carry_flag_after"] is True
    assert rows[2]["0x12da"]["overflow_flag_after"] is True
    assert rows[3]["0x12eb"]["registers_after"]["rax"] == "0x1234"
    assert rows[3]["0x12eb"]["carry_flag_after"] is True
    assert rows[4]["0x12fa"]["registers_after"]["rax"] == "?"
    assert rows[4]["0x12fa"]["carry_flag_after"] is None


def test_memory_xadd_and_cmpxchg_apply_known_single_thread_transfers():
    xadd_block = block(0x12E0, [
        ins(0x12E0, "mov", "dword ptr [rbp - 4], 7"),
        ins(0x12E1, "mov", "eax, 5"),
        ins(0x12E2, "xadd", "dword ptr [rbp - 4], eax"),
        ins(0x12E3, "mov", "ecx, dword ptr [rbp - 4]"),
    ])
    xadd_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x12E0, [xadd_block])
    xadd_by_address = {row["address"]: row for row in xadd_rows}
    assert xadd_by_address["0x12e2"]["registers_after"]["rax"] == "7"
    assert xadd_by_address["0x12e3"]["write"]["value"] == "0xc"

    cmpxchg_success = block(0x12F0, [
        ins(0x12F0, "mov", "dword ptr [rbp - 4], 7"),
        ins(0x12F1, "mov", "eax, 7"),
        ins(0x12F2, "mov", "ecx, 9"),
        ins(0x12F3, "cmpxchg", "dword ptr [rbp - 4], ecx"),
        ins(0x12F4, "mov", "edx, dword ptr [rbp - 4]"),
    ])
    success_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x12F0, [cmpxchg_success])
    success_by_address = {row["address"]: row for row in success_rows}
    assert success_by_address["0x12f3"]["zero_flag_after"] is True
    assert success_by_address["0x12f4"]["write"]["value"] == "9"

    cmpxchg_failure = block(0x1300, [
        ins(0x1300, "mov", "dword ptr [rbp - 4], 7"),
        ins(0x1301, "mov", "eax, 5"),
        ins(0x1302, "mov", "ecx, 9"),
        ins(0x1303, "cmpxchg", "dword ptr [rbp - 4], ecx"),
        ins(0x1304, "mov", "edx, dword ptr [rbp - 4]"),
    ])
    failure_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x1300, [cmpxchg_failure])
    failure_by_address = {row["address"]: row for row in failure_rows}
    assert failure_by_address["0x1303"]["registers_after"]["rax"] == "7"
    assert failure_by_address["0x1303"]["zero_flag_after"] is False
    assert failure_by_address["0x1304"]["write"]["value"] == "7"


def test_direction_flag_drives_string_pointer_updates_and_merges_conservatively():
    forward = block(0x12E0, [
        ins(0x12E0, "mov", "esi, 0x1000"),
        ins(0x12E1, "mov", "edi, 0x2000"),
        ins(0x12E2, "mov", "ecx, 3"),
        ins(0x12E3, "cld"),
        ins(0x12E4, "rep movsb"),
        ins(0x12E5, "ret", flow="return"),
    ])
    backward = block(0x12F0, [
        ins(0x12F0, "mov", "esi, 0x1000"),
        ins(0x12F1, "mov", "edi, 0x2000"),
        ins(0x12F2, "mov", "ecx, 2"),
        ins(0x12F3, "std"),
        ins(0x12F4, "rep movsw"),
        ins(0x12F5, "ret", flow="return"),
    ])
    forward_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x12E0, [forward])
    backward_rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x12F0, [backward])
    forward_move = next(row for row in forward_rows if row["address"] == "0x12e4")
    backward_move = next(row for row in backward_rows if row["address"] == "0x12f4")
    assert forward_move["registers_after"]["rsi"] == "0x1003"
    assert forward_move["registers_after"]["rdi"] == "0x2003"
    assert forward_move["direction_flag_after"] is False
    assert backward_move["registers_after"]["rsi"] == "0xffc"
    assert backward_move["registers_after"]["rdi"] == "0x1ffc"
    assert backward_move["direction_flag_after"] is True
    assert AbstractState.merge(AbstractState(direction=False), AbstractState(direction=True)).direction is None


def test_dataflow_tracks_types_through_stack_and_call_result():
    b = block(0x3500, [
        ins(0x3500, "lea", "rcx, [rip + 0x20]", ref=0x4020),
        ins(0x3501, "mov", "dword ptr [rbp - 4], 6"),
        ins(0x3502, "mov", "eax, dword ptr [rbp - 4]"),
        ins(0x3503, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(64, {}, {}, {})
    rows, _ = flow.analyze(0x3500, [b])
    address = next(row for row in rows if row["address"] == "0x3500")
    store = next(row for row in rows if row["address"] == "0x3501")
    load = next(row for row in rows if row["address"] == "0x3502")
    assert address["write_type"] == "uintptr_t"
    assert store["write_type"] == "uint32_t"
    assert load["register_types_after"]["rax"] == "uint32_t"


def test_dataflow_tracks_non_stack_memory_aliases_conservatively():
    b = block(0x3600, [
        ins(0x3600, "mov", "dword ptr [rax + 4], 7"),
        ins(0x3601, "mov", "ecx, dword ptr [rax + 4]"),
        ins(0x3602, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(64, {}, {}, {})
    rows, _ = flow.analyze(0x3600, [b])
    load = next(row for row in rows if row["address"] == "0x3601")
    assert load["resolved_operands"][1] == "7"
    assert "rax + 4" in load["memory_after"]


def test_memory_aliases_track_equal_concrete_pointer_bases_and_offsets():
    b = block(0x3620, [
        ins(0x3620, "mov", "rax, 0x1000"),
        ins(0x3621, "mov", "rbx, rax"),
        ins(0x3622, "mov", "dword ptr [rax + 4], 7"),
        ins(0x3623, "mov", "ecx, dword ptr [rbx + 0x4]"),
        ins(0x3624, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x3620, [b])
    load = next(row for row in rows if row["address"] == "0x3623")
    assert load["resolved_operands"][1] == "7"
    assert load["memory_after"] == {"va:0x1004": "7"}


def test_memory_aliases_follow_register_copies_of_symbolic_pointer_arguments():
    b = block(0x3630, [
        ins(0x3630, "mov", "rax, rcx"),
        ins(0x3631, "add", "rax, 4"),
        ins(0x3632, "mov", "dword ptr [rax], 9"),
        ins(0x3633, "mov", "edx, dword ptr [rcx + 0x4]"),
        ins(0x3634, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x3630, [b])
    load = next(row for row in rows if row["address"] == "0x3633")
    assert load["resolved_operands"][1] == "9"
    assert load["memory_after"] == {"symbol:arg1+4": "9"}


def test_lea_propagates_canonical_symbolic_pointer_to_later_dereference():
    b = block(0x3635, [
        ins(0x3635, "lea", "rbx, [rcx + 4]"),
        ins(0x3636, "mov", "dword ptr [rbx], 9"),
        ins(0x3637, "mov", "eax, dword ptr [rcx + 4]"),
        ins(0x3638, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x3635, [b])
    lea = next(row for row in rows if row["address"] == "0x3635")
    store = next(row for row in rows if row["address"] == "0x3636")
    load = next(row for row in rows if row["address"] == "0x3637")
    assert lea["registers_after"]["rbx"] == "symbol:arg1+4"
    assert store["memory_after"] == {"symbol:arg1+4": "9"}
    assert load["resolved_operands"][1] == "9"


def test_lea_propagates_symbolic_scaled_index_address_to_direct_access():
    b = block(0x3639, [
        ins(0x3639, "mov", "rax, rcx"),
        ins(0x363A, "lea", "rbx, [rax + rdx*4 + 8]"),
        ins(0x363B, "mov", "dword ptr [rbx], 15"),
        ins(0x363C, "mov", "r8d, dword ptr [rcx + 8 + rdx*4]"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x3639, [b])
    store = next(row for row in rows if row["address"] == "0x363b")
    load = next(row for row in rows if row["address"] == "0x363c")
    assert len(store["memory_after"]) == 1
    assert load["resolved_operands"][1] == "0xf"


def test_lea_propagates_multiple_symbolic_index_address():
    b = block(0x363D, [
        ins(0x363D, "mov", "rax, rcx"),
        ins(0x363E, "lea", "rbx, [rax + rdx*4 + r8*8 + 12]"),
        ins(0x363F, "mov", "dword ptr [rbx], 17"),
        ins(0x3640, "mov", "r9d, dword ptr [rcx + 0xc + r8*8 + rdx*4]"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x363D, [b])
    store = next(row for row in rows if row["address"] == "0x363f")
    load = next(row for row in rows if row["address"] == "0x3640")
    assert len(store["memory_after"]) == 1
    assert load["resolved_operands"][1] == "0x11"


def test_nonstack_memory_writes_invalidate_overlaps_but_preserve_disjoint_ranges():
    disjoint = block(0x3638, [
        ins(0x3638, "mov", "rax, rcx"),
        ins(0x3639, "mov", "dword ptr [rcx], 0x11223344"),
        ins(0x363A, "mov", "dword ptr [rax + 8], 0x55667788"),
        ins(0x363B, "mov", "r8d, dword ptr [rcx]"),
        ins(0x363C, "mov", "r9d, dword ptr [rax + 8]"),
        ins(0x363D, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x3638, [disjoint])
    low = next(row for row in rows if row["address"] == "0x363b")
    high = next(row for row in rows if row["address"] == "0x363c")
    assert low["resolved_operands"][1] == "0x11223344"
    assert high["resolved_operands"][1] == "0x55667788"

    overlapping = block(0x3648, [
        ins(0x3648, "mov", "rax, rcx"),
        ins(0x3649, "mov", "dword ptr [rcx], 0x11223344"),
        ins(0x364A, "mov", "word ptr [rax + 2], 0x1234"),
        ins(0x364B, "mov", "r8d, dword ptr [rcx]"),
        ins(0x364C, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x3648, [overlapping])
    load = next(row for row in rows if row["address"] == "0x364b")
    assert load["resolved_operands"][1] != "0x11223344"


def test_nonstack_memory_fact_requires_matching_access_width():
    b = block(0x3640, [
        ins(0x3640, "mov", "rax, 0x1000"),
        ins(0x3641, "mov", "dword ptr [rax + 4], 7"),
        ins(0x3642, "mov", "rcx, qword ptr [rax + 4]"),
        ins(0x3643, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x3640, [b])
    store = next(row for row in rows if row["address"] == "0x3641")
    load = next(row for row in rows if row["address"] == "0x3642")
    assert store["memory_widths_after"] == {"va:0x1004": 4}
    assert load["resolved_operands"][1] == "?"
    assert load["registers_after"]["rcx"] == "?"


def test_scaled_array_aliases_fold_concrete_base_index_and_displacement():
    b = block(0x3650, [
        ins(0x3650, "mov", "rax, 0x1000"),
        ins(0x3651, "mov", "rbx, rax"),
        ins(0x3652, "mov", "ecx, 2"),
        ins(0x3653, "mov", "dword ptr [rax + rcx*4 + 8], 7"),
        ins(0x3654, "mov", "edx, dword ptr [rbx + 8 + rcx*4]"),
        ins(0x3655, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x3650, [b])
    store = next(row for row in rows if row["address"] == "0x3653")
    load = next(row for row in rows if row["address"] == "0x3654")
    assert store["memory_after"] == {"va:0x1010": "7"}
    assert load["resolved_operands"][1] == "7"


def test_scaled_array_aliases_follow_symbolic_index_register_copies():
    b = block(0x3660, [
        ins(0x3660, "mov", "rax, rcx"),
        ins(0x3661, "mov", "rbx, rax"),
        ins(0x3662, "mov", "r8, rdx"),
        ins(0x3663, "mov", "dword ptr [rax + rdx*4 + 8], 9"),
        ins(0x3664, "mov", "ecx, dword ptr [rbx + r8*4 + 0x8]"),
        ins(0x3665, "ret", flow="return"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x3660, [b])
    store = next(row for row in rows if row["address"] == "0x3663")
    load = next(row for row in rows if row["address"] == "0x3664")
    assert len(store["memory_after"]) == 1
    assert load["resolved_operands"][1] == "9"


def test_multiple_scaled_array_indices_canonicalize_order_and_concrete_offsets():
    b = block(0x3670, [
        ins(0x3670, "mov", "rax, 0x1000"),
        ins(0x3671, "mov", "rbx, rax"),
        ins(0x3672, "mov", "ecx, 2"),
        ins(0x3673, "mov", "edx, 3"),
        ins(0x3674, "mov", "dword ptr [rax + rcx*4 + rdx*8 + 0x10], 11"),
        ins(0x3675, "mov", "r8d, dword ptr [rbx + 0x10 + rdx*8 + rcx*4]"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x3670, [b])
    store = next(row for row in rows if row["address"] == "0x3674")
    load = next(row for row in rows if row["address"] == "0x3675")
    assert store["memory_after"] == {"va:0x1030": "0xb"}
    assert load["resolved_operands"][1] == "0xb"


def test_multiple_scaled_array_indices_follow_symbolic_copies():
    b = block(0x3680, [
        ins(0x3680, "mov", "rax, rcx"),
        ins(0x3681, "mov", "rbx, rax"),
        ins(0x3682, "mov", "r10, rdx"),
        ins(0x3683, "mov", "r11, r8"),
        ins(0x3684, "mov", "dword ptr [rax + rdx*4 + r8*8 + 12], 13"),
        ins(0x3685, "mov", "r9d, dword ptr [rbx + 0xc + r11*8 + r10*4]"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x3680, [b])
    store = next(row for row in rows if row["address"] == "0x3684")
    load = next(row for row in rows if row["address"] == "0x3685")
    assert len(store["memory_after"]) == 1
    assert load["resolved_operands"][1] == "0xd"


def test_memory_keys_canonicalize_equivalent_numeric_offset_formatting():
    b = block(0x3650, [
        ins(0x3650, "mov", "dword ptr [rax + 0x4], 7"),
        ins(0x3651, "mov", "ecx, dword ptr [rax+4]"),
        ins(0x3652, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(64, {}, {}, {})
    rows, _ = flow.analyze(0x3650, [b])
    load = next(row for row in rows if row["address"] == "0x3651")
    assert load["resolved_operands"][1] == "7"
    assert "rax + 4" in load["memory_after"]


def test_unknown_call_invalidates_non_stack_memory_aliases():
    b = block(0x3700, [
        ins(0x3700, "mov", "dword ptr [rax + 4], 7"),
        ins(0x3701, "call", "rax", flow="call"),
        ins(0x3702, "mov", "ecx, dword ptr [rax + 4]"),
        ins(0x3703, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(64, {}, {}, {})
    rows, _ = flow.analyze(0x3700, [b])
    load = next(row for row in rows if row["address"] == "0x3702")
    assert load["resolved_operands"][1] == "mem[rax + 4]"
    assert load["memory_after"] == {}


def test_indirect_call_through_propagated_function_address_is_resolved():
    caller = block(0x3800, [
        ins(0x3800, "mov", "rax, 0x4000"),
        ins(0x3801, "mov", "ecx, 5"),
        ins(0x3802, "call", "rax", flow="call"),
        ins(0x3803, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(
        64, {}, {}, {0x4000: "callee"},
        function_arg_counts={0x4000: 1},
        function_return_types={0x4000: "uint32_t"},
        known_code_addresses={0x4000},
    )
    rows, _ = flow.analyze(0x3800, [caller])
    call = next(row for row in rows if row["address"] == "0x3802")
    assert call["resolved_call_target"] == "0x4000"
    assert call["write"]["value"] == "callee(5)"
    assert call["write_type"] == "uint32_t"


def test_x86_resolved_indirect_call_uses_known_callee_arity_and_pushes():
    caller = block(0x3810, [
        ins(0x3810, "mov", "eax, 0x4000"),
        ins(0x3811, "push", "7"),
        ins(0x3812, "call", "eax", flow="call"),
    ])
    flow = FunctionDataFlow(
        32, {}, {}, {0x4000: "callee"}, function_arg_counts={0x4000: 2},
        known_code_addresses={0x4000},
    )
    rows, _ = flow.analyze(0x3810, [caller])
    call = next(row for row in rows if row["address"] == "0x3812")
    assert call["resolved_call_target"] == "0x4000"
    assert call["call_argument_values"] == ["7"]
    assert call["write"]["value"] == "callee(7, unknown_stack_arg_2)"


def test_indirect_call_through_import_address_table_load_is_named():
    caller = block(0x3900, [
        Instruction(0x3900, 1, "90", "mov", "rax, qword ptr [rip + 0x10]", ".text",
                    reference_target=0x5000),
        ins(0x3901, "mov", "qword ptr [rbp - 8], rax"),
        ins(0x3902, "mov", "rcx, qword ptr [rbp - 8]"),
        ins(0x3903, "call", "rcx", flow="call"),
        ins(0x3904, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(64, {}, {0x5000: "KERNEL32.dll!GetLastError"}, {})
    rows, _ = flow.analyze(0x3900, [caller])
    call = next(row for row in rows if row["address"] == "0x3903")
    assert call["resolved_operands"][0] == "iat(KERNEL32.dll!GetLastError)"
    assert call["resolved_call_import"] == "KERNEL32.dll!GetLastError"


def test_indirect_call_through_static_vtable_slot_resolves_code_target():
    caller = block(0x3A00, [
        Instruction(0x3A00, 1, "90", "mov", "rax, qword ptr [rip + 0x10]", ".text",
                    reference_target=0x6000),
        ins(0x3A01, "call", "qword ptr [rax + 8]", flow="call"),
        ins(0x3A02, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(
        64, {}, {}, {0x4010: "virtual_method"}, known_code_addresses={0x4010},
        static_pointer_slots={0x6000: 0x7000, 0x7008: 0x4010},
    )
    rows, _ = flow.analyze(0x3A00, [caller])
    call = next(row for row in rows if row["address"] == "0x3a01")
    assert call["resolved_operands"][0] == "static_ptr(0x4010)"
    assert call["resolved_call_target"] == "0x4010"
    assert call["resolved_call_provenance"] == "static_pointer_slot"


def test_indirect_call_through_static_object_vptr_resolves_virtual_method():
    caller = block(0x3A10, [
        Instruction(0x3A10, 1, "90", "lea", "rcx, [rip + 0x20]", ".text",
                    reference_target=0x6100),
        ins(0x3A11, "mov", "rax, qword ptr [rcx]"),
        ins(0x3A12, "call", "qword ptr [rax + 0x10]", flow="call"),
        ins(0x3A13, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(
        64, {}, {}, {0x4010: "virtual_method"}, known_code_addresses={0x4010},
        static_pointer_slots={0x6100: 0x7000, 0x7010: 0x4010},
    )
    rows, _ = flow.analyze(0x3A10, [caller])
    virtual_call = next(row for row in rows if row["address"] == "0x3a12")
    assert virtual_call["resolved_operands"][0] == "static_ptr(0x4010)"
    assert virtual_call["resolved_call_target"] == "0x4010"
    assert virtual_call["resolved_call_provenance"] == "static_pointer_slot"


def test_indirect_call_constant_that_is_not_code_is_not_promoted():
    caller = block(0x3900, [
        ins(0x3900, "mov", "rax, 0xdead"),
        ins(0x3901, "call", "rax", flow="call"),
        ins(0x3902, "ret", flow="return"),
    ])
    flow = FunctionDataFlow(64, {}, {}, {}, known_code_addresses={0x4000})
    rows, _ = flow.analyze(0x3900, [caller])
    call = next(row for row in rows if row["address"] == "0x3901")
    assert call["resolved_call_target"] is None
    assert call["write"]["value"].startswith("indirect_call(")


def test_unknown_loop_values_are_bounded_by_widening_join():
    entry = block(0x5000, [ins(0x5000, "mov", "eax, 1"), ins(0x5001, "jmp", "0x5002", target=0x5002, flow="jump")], [0x5002])
    loop = block(0x5002, [ins(0x5002, "add", "eax, 1"), ins(0x5003, "jmp", "0x5002", target=0x5002, flow="jump")], [0x5002])
    flow = FunctionDataFlow(64, {}, {}, {}, max_steps_per_function=200)
    rows, summary = flow.analyze(0x5000, [entry, loop])
    assert summary["analysis_steps"] < 200
    assert len(rows) <= 4


def test_memory_fences_preserve_single_thread_register_and_memory_facts():
    b = block(0x5100, [
        ins(0x5100, "mov", "eax, 5"),
        ins(0x5101, "mov", "dword ptr [0x2000], 7"),
        ins(0x5102, "mfence"),
        ins(0x5103, "mov", "ebx, eax"),
        ins(0x5104, "mov", "ecx, dword ptr [0x2000]"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x5100, [b])
    by_address = {row["address"]: row for row in rows}
    fence = by_address["0x5102"]
    assert fence["memory_ordering_fence"] == "full"
    assert fence["registers_after"]["rax"] == "5"
    assert fence["memory_after"]["va:0x2000"] == "7"
    assert by_address["0x5103"]["write"]["value"] == "5"
    assert by_address["0x5104"]["write"]["value"] == "7"


def test_sse_float_compare_sets_ordered_and_unordered_flags():
    b = block(0x5200, [
        ins(0x5200, "mov", "eax, 0x3f800000"),
        ins(0x5201, "movd", "xmm0, eax"),
        ins(0x5202, "mov", "ebx, 0x40000000"),
        ins(0x5203, "movd", "xmm1, ebx"),
        ins(0x5204, "ucomiss", "xmm0, xmm1"),
        ins(0x5205, "jb", "0x5210", target=0x5210, flow="conditional"),
        ins(0x5206, "mov", "eax, 0x7fc00000"),
        ins(0x5207, "movd", "xmm0, eax"),
        ins(0x5208, "comiss", "xmm0, xmm1"),
        ins(0x5209, "jp", "0x5210", target=0x5210, flow="conditional"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x5200, [b])
    by_address = {row["address"]: row for row in rows}
    ordered = by_address["0x5204"]
    unordered = by_address["0x5208"]
    assert ordered["carry_flag_after"] is True
    assert ordered["zero_flag_after"] is False
    assert ordered["parity_flag_after"] is False
    assert by_address["0x5205"]["condition"] == "1"
    assert unordered["carry_flag_after"] is True
    assert unordered["zero_flag_after"] is True
    assert unordered["parity_flag_after"] is True
    assert by_address["0x5209"]["condition"] == "1"


def test_scalar_sse_float_arithmetic_folds_only_exact_finite_results():
    b = block(0x5300, [
        ins(0x5300, "mov", "eax, 0x3fc00000"),
        ins(0x5301, "movd", "xmm0, eax"),
        ins(0x5302, "mov", "ebx, 0x40100000"),
        ins(0x5303, "movd", "xmm1, ebx"),
        ins(0x5304, "addss", "xmm0, xmm1"),
        ins(0x5305, "movd", "ecx, xmm0"),
        ins(0x5306, "mov", "edx, 0x3f800000"),
        ins(0x5307, "movd", "xmm2, edx"),
        ins(0x5308, "mov", "edx, 0x40100000"),
        ins(0x5309, "pinsrd", "xmm2, edx, 1"),
        ins(0x530A, "vaddss", "xmm3, xmm2, xmm1"),
        ins(0x530B, "mov", "eax, 0x33800000"),
        ins(0x530C, "movd", "xmm4, eax"),
        ins(0x530D, "mov", "eax, 0x3f800000"),
        ins(0x530E, "movd", "xmm5, eax"),
        ins(0x530F, "addss", "xmm4, xmm5"),
        ins(0x5310, "mov", "rax, 0x3ff8000000000000"),
        ins(0x5311, "movq", "xmm6, rax"),
        ins(0x5312, "mov", "rbx, 0x4002000000000000"),
        ins(0x5313, "movq", "xmm7, rbx"),
        ins(0x5314, "addsd", "xmm6, xmm7"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x5300, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x5304"]["write"]["value"] == "0x40700000"
    assert by_address["0x5305"]["write"]["value"] == "0x40700000"
    assert by_address["0x530a"]["write"]["value"] == "0x4010000040500000"
    assert by_address["0x530f"]["write"]["value"] == "?"
    assert by_address["0x5314"]["write"]["value"] == "0x400e000000000000"


def test_scalar_sse_minmax_preserves_second_source_nan_and_zero_bits():
    b = block(0x5400, [
        ins(0x5400, "mov", "eax, 0x7fc12345"),
        ins(0x5401, "movd", "xmm0, eax"),
        ins(0x5402, "mov", "ebx, 0x40100000"),
        ins(0x5403, "movd", "xmm1, ebx"),
        ins(0x5404, "minss", "xmm0, xmm1"),
        ins(0x5405, "mov", "eax, 0x80000000"),
        ins(0x5406, "movd", "xmm2, eax"),
        ins(0x5407, "mov", "ebx, 0"),
        ins(0x5408, "movd", "xmm3, ebx"),
        ins(0x5409, "maxss", "xmm2, xmm3"),
        ins(0x540A, "mov", "eax, 0x7fc12345"),
        ins(0x540B, "movd", "xmm0, eax"),
        ins(0x540C, "vminss", "xmm4, xmm1, xmm0"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x5400, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x5404"]["write"]["value"] == "0x40100000"
    assert by_address["0x5409"]["write"]["value"] == "0"
    assert by_address["0x540c"]["write"]["value"] == "0x7fc12345"


def test_sse_integer_float_conversions_fold_only_defined_cases():
    b = block(0x5500, [
        ins(0x5500, "mov", "eax, 0x1000000"),
        ins(0x5501, "movd", "xmm0, eax"),
        ins(0x5502, "cvtsi2ss", "xmm0, eax"),
        ins(0x5503, "movd", "ebx, xmm0"),
        ins(0x5504, "mov", "eax, 0x1000001"),
        ins(0x5505, "cvtsi2ss", "xmm1, eax"),
        ins(0x5506, "movd", "ecx, xmm1"),
        ins(0x5507, "mov", "eax, 0x40100000"),
        ins(0x5508, "movd", "xmm2, eax"),
        ins(0x5509, "cvttss2si", "edx, xmm2"),
        ins(0x550A, "cvtss2si", "r8d, xmm2"),
        ins(0x550B, "mov", "eax, 0xbfe00000"),
        ins(0x550C, "movd", "xmm3, eax"),
        ins(0x550D, "cvttss2si", "r9d, xmm3"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x5500, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x5503"]["write"]["value"] == "0x4b800000"
    assert by_address["0x5506"]["write"]["value"] == "?"
    assert by_address["0x5509"]["write"]["value"] == "2"
    assert by_address["0x550a"]["write"]["value"] == "?"
    assert by_address["0x550d"]["write"]["value"] == "0xffffffff"


def test_x87_abs_and_sign_change_update_only_top_stack_value():
    b = block(0x5600, [
        ins(0x5600, "fld1"),
        ins(0x5601, "fchs"),
        ins(0x5602, "fld1"),
        ins(0x5603, "fchs"),
        ins(0x5604, "fabs"),
    ])
    rows, _ = FunctionDataFlow(64, {}, {}, {}).analyze(0x5600, [b])
    by_address = {row["address"]: row for row in rows}
    assert by_address["0x5601"]["x87_stack_after"] == ["-1.0"]
    assert by_address["0x5603"]["x87_stack_after"] == ["-1.0", "-1.0"]
    assert by_address["0x5604"]["x87_stack_after"] == ["1.0", "-1.0"]
