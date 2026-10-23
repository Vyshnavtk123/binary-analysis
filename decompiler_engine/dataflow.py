from __future__ import annotations

"""Bounded forward symbolic data-flow for the independent PE analysis engine.

This is a conservative abstract interpreter, not a complete x86 semantics engine.
It tracks simple constants/expressions in general-purpose registers and stack-relative
memory, propagates values between CFG blocks, and widens unstable loop values to unknown.
"""

import json
import math
import re
import struct
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .api_signatures import API_PARAMETER_TYPES, get_api_signature, normalize_api_name
from .models import BasicBlock, Instruction
from .ir import lift_instruction

_UNKNOWN = "?"
_STRING_KINDS = {
    **{f"{stem}{suffix}": (kind, width) for stem, kind in (
        ("movs", "move"), ("cmps", "compare"), ("stos", "store"),
        ("lods", "load"), ("scas", "scan"), ("ins", "input"), ("outs", "output"),
    ) for suffix, width in (("b", 8), ("w", 16), ("d", 32))},
    "movsq": ("move", 64), "cmpsq": ("compare", 64), "stosq": ("store", 64),
    "lodsq": ("load", 64), "scasq": ("scan", 64),
}
_REPEAT_PREFIXES = {"rep", "repe", "repz", "repne", "repnz"}
_TYPE_UNKNOWN = "unknown"
_REG_FAMILY = {
    # Legacy x86/x64 general-purpose register aliases.
    "rax": "rax", "eax": "rax", "ax": "rax", "al": "rax", "ah": "rax",
    "rbx": "rbx", "ebx": "rbx", "bx": "rbx", "bl": "rbx", "bh": "rbx",
    "rcx": "rcx", "ecx": "rcx", "cx": "rcx", "cl": "rcx", "ch": "rcx",
    "rdx": "rdx", "edx": "rdx", "dx": "rdx", "dl": "rdx", "dh": "rdx",
    "rsi": "rsi", "esi": "rsi", "si": "rsi", "sil": "rsi",
    "rdi": "rdi", "edi": "rdi", "di": "rdi", "dil": "rdi",
    "rbp": "rbp", "ebp": "rbp", "bp": "rbp", "bpl": "rbp",
    "rsp": "rsp", "esp": "rsp", "sp": "rsp", "spl": "rsp",
    "rip": "rip", "eip": "rip", "ip": "rip",
}
for _i in range(8, 16):
    _REG_FAMILY.update({f"r{_i}": f"r{_i}", f"r{_i}d": f"r{_i}",
                        f"r{_i}w": f"r{_i}", f"r{_i}b": f"r{_i}"})
for _i in range(32):
    _REG_FAMILY.update({f"xmm{_i}": f"xmm{_i}", f"ymm{_i}": f"ymm{_i}",
                        f"zmm{_i}": f"zmm{_i}"})

_BINARY_OPS = {
    "add": "+", "sub": "-", "imul": "*", "mul": "*", "and": "&",
    "or": "|", "xor": "^", "shl": "<<", "sal": "<<", "shr": ">>", "sar": ">>",
}
_BRANCH_REL = {
    "je": "==", "jz": "==", "jne": "!=", "jnz": "!=",
    "jg": ">", "jnle": ">", "jge": ">=", "jnl": ">=",
    "jl": "<", "jnge": "<", "jle": "<=", "jng": "<=",
    "ja": ">", "jnbe": ">", "jae": ">=", "jnb": ">=", "jnc": ">=",
    "jb": "<", "jnae": "<", "jc": "<", "jbe": "<=", "jna": "<=",
}
_UNSIGNED_BRANCHES = {"ja", "jnbe", "jae", "jnb", "jnc", "jb", "jnae", "jc", "jbe", "jna"}


def split_operands(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _canon_reg(reg: str) -> str | None:
    return _REG_FAMILY.get(reg.strip().lower())


def _register_slice(reg: str, bits: int) -> tuple[str, int, int] | None:
    """Return (family, width, bit offset) for a scalar register alias."""
    token = reg.strip().lower()
    family = _canon_reg(token)
    if family is None or family.startswith(("xmm", "ymm", "zmm")):
        return None
    if token in {"ah", "bh", "ch", "dh"}:
        return family, 8, 8
    if token in {"al", "bl", "cl", "dl", "sil", "dil", "bpl", "spl"} or re.fullmatch(r"r(?:8|9|1[0-5])b", token):
        return family, 8, 0
    if token in {"ax", "bx", "cx", "dx", "si", "di", "bp", "sp"} or re.fullmatch(r"r(?:8|9|1[0-5])w", token):
        return family, 16, 0
    if token in {"eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp", "eip"} or re.fullmatch(r"r(?:8|9|1[0-5])d", token):
        return family, 32, 0
    return family, bits, 0


def _mem_inner(text: str) -> str | None:
    match = re.match(r"\s*(?:(?:byte|word|dword|qword|xmmword|ymmword|zmmword)\s+ptr\s+)?\[(.+)\]\s*$", text, re.I)
    return re.sub(r"\s+", " ", match.group(1).strip().lower()) if match else None


def _canonical_mem_inner(inner: str | None) -> str | None:
    if inner is None:
        return None
    # Normalize stack-base aliases inherited from the original model and
    # superficial numeric formatting only. Keep general register widths distinct:
    # in x64, an address-size override can make [eax] differ from [rax].
    canonical = re.sub(r"\b(?:ebp|bp)\b", "rbp", re.sub(r"\b(?:esp|sp)\b", "rsp", inner.lower()))

    def normalize_integer(match: re.Match[str]) -> str:
        try:
            return str(int(match.group(0), 0))
        except ValueError:
            return match.group(0).lower()

    canonical = re.sub(
        r"(?<![A-Za-z0-9_])(?:0x[0-9a-f]+|\d+)(?![A-Za-z0-9_])",
        normalize_integer,
        canonical,
        flags=re.I,
    )
    canonical = re.sub(r"\s*([+*\-])\s*", r"\1", canonical)
    canonical = re.sub(r"\s+", " ", canonical).strip()
    stack_offset = re.fullmatch(r"(rbp|rsp)([+-])(\d+)", canonical)
    if stack_offset:
        return f"{stack_offset.group(1)} {stack_offset.group(2)} {stack_offset.group(3)}"
    return re.sub(r"\s*([+*\-])\s*", r" \1 ", canonical)


def _stack_argument_name(inner: str | None) -> str | None:
    if inner is None:
        return None
    match = re.fullmatch(r"(?:rbp|rsp)\s*\+\s*(0x[0-9a-f]+|\d+)", inner)
    if not match:
        return None
    offset = _parse_int(match.group(1))
    return f"stack_arg_{offset:X}" if offset is not None else None


def _is_stack_mem(inner: str | None) -> bool:
    if inner is None:
        return False
    return bool(re.search(r"\b(?:rbp|ebp|rsp|esp)\b", inner)) and "rip" not in inner


def _stack_address(inner: str | None) -> tuple[str, int] | None:
    if inner is None:
        return None
    match = re.fullmatch(r"(rbp|rsp)(?:\s*([+-])\s*(0x[0-9a-f]+|\d+))?", inner, re.I)
    if not match:
        return None
    base, sign, offset_text = match.groups()
    offset = _parse_int(offset_text) if offset_text else 0
    if offset is None:
        return None
    return base.lower(), -offset if sign == "-" else offset


def _operand_width_bytes(operand: str, bits: int) -> int:
    size = re.match(r"\s*(byte|word|dword|qword|xmmword|ymmword|zmmword)\s+ptr\b", operand, re.I)
    if size:
        return {"byte": 1, "word": 2, "dword": 4, "qword": 8,
                "xmmword": 16, "ymmword": 32, "zmmword": 64}[size.group(1).lower()]
    token = operand.strip().lower()
    if token in {"al", "ah", "bl", "bh", "cl", "ch", "dl", "dh", "sil", "dil", "bpl", "spl"} or re.fullmatch(r"r(?:8|9|1[0-5])b", token):
        return 1
    if token in {"ax", "bx", "cx", "dx", "si", "di", "bp", "sp"} or re.fullmatch(r"r(?:8|9|1[0-5])w", token):
        return 2
    if token in {"eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp"} or re.fullmatch(r"r(?:8|9|1[0-5])d", token):
        return 4
    vector = re.fullmatch(r"([xyz])mm(?:[0-9]|[12][0-9]|3[01])", token)
    if vector:
        return {"x": 16, "y": 32, "z": 64}[vector.group(1)]
    return bits // 8


def _parse_int(text: str) -> int | None:
    s = text.strip().lower()
    try:
        if re.fullmatch(r"[-+]?0x[0-9a-f]+", s):
            return int(s, 16)
        if re.fullmatch(r"[-+]?\d+", s):
            return int(s, 10)
    except ValueError:
        pass
    return None


def _format_int(value: int) -> str:
    return str(value) if -9 <= value <= 9 else (str(value) if value < 0 else hex(value))


def _fold(op: str, lhs: str, rhs: str) -> str:
    a, b = _parse_int(lhs), _parse_int(rhs)
    if a is not None and b is not None:
        try:
            result = {
                "+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
                "&": lambda: a & b, "|": lambda: a | b, "^": lambda: a ^ b,
                "<<": lambda: a << b if 0 <= b < 128 else None,
                ">>": lambda: a >> b if 0 <= b < 128 else None,
            }[op]()
            if result is not None:
                return _format_int(result)
        except (OverflowError, ValueError, KeyError):
            pass
    if op == "+" and b == 0:
        return lhs
    if op == "+" and a == 0:
        return rhs
    if op == "-" and b == 0:
        return lhs
    if op == "*" and (a == 0 or b == 0):
        return "0"
    if op == "*" and a == 1:
        return rhs
    if op == "*" and b == 1:
        return lhs
    expr = f"({lhs} {op} {rhs})"
    return expr if len(expr) <= 160 else _UNKNOWN


def _fold_alu(mnemonic: str, op: str, lhs: str, rhs: str, width_bits: int) -> str:
    """Fold integer ALU operations using the destination's x86 operand width."""
    if lhs == _UNKNOWN or rhs == _UNKNOWN:
        return _UNKNOWN
    left, right = _parse_int(lhs), _parse_int(rhs)
    if left is None or right is None or width_bits <= 0:
        return _fold(op, lhs, rhs)
    mask = (1 << width_bits) - 1
    left &= mask
    if mnemonic in {"shl", "sal", "shr", "sar"}:
        # Intel masks the count to five bits for operands up to 32 bits and
        # six bits for 64-bit operands. A masked count of zero preserves value.
        count = right & (0x3F if width_bits == 64 else 0x1F)
        if count == 0:
            result = left
        elif mnemonic == "sar":
            sign = 1 << (width_bits - 1)
            signed_left = left - (1 << width_bits) if left & sign else left
            result = signed_left >> count
        elif mnemonic in {"shl", "sal"}:
            result = left << count
        else:
            result = left >> count
    else:
        result = {
            "+": lambda: left + right,
            "-": lambda: left - right,
            "*": lambda: left * right,
            "&": lambda: left & right,
            "|": lambda: left | right,
            "^": lambda: left ^ right,
        }[op]()
    return _format_int(result & mask)


def _fold_unary_alu(mnemonic: str, value: str, width_bits: int) -> str:
    if value == _UNKNOWN:
        return _UNKNOWN
    number = _parse_int(value)
    if number is None or width_bits <= 0:
        if mnemonic in {"inc", "dec"}:
            return _fold("+" if mnemonic == "inc" else "-", value, "1")
        expression = f"(-{value})"
        return expression if len(expression) <= 160 else _UNKNOWN
    mask = (1 << width_bits) - 1
    number &= mask
    result = number + 1 if mnemonic == "inc" else number - 1 if mnemonic == "dec" else -number
    return _format_int(result & mask)


def _fold_packed_integer(mnemonic: str, left: str, right: str, vector_bits: int) -> str:
    """Fold common packed integer SIMD operations when both vectors are concrete."""
    if left == right and (mnemonic in {"pxor", "vpxor"}
                          or re.fullmatch(r"v?psub[bwdq]", mnemonic)):
        return "0"
    a, b = _parse_int(left), _parse_int(right)
    if a is None or b is None or vector_bits not in {128, 256, 512}:
        return _UNKNOWN
    mask = (1 << vector_bits) - 1
    if mnemonic in {"pxor", "vpxor"}:
        return _format_int((a ^ b) & mask)
    if mnemonic in {"por", "vpor"}:
        return _format_int((a | b) & mask)
    if mnemonic in {"pand", "vpand"}:
        return _format_int((a & b) & mask)
    if mnemonic in {"pandn", "vpandn"}:
        return _format_int((~a & b) & mask)
    unpack = re.fullmatch(r"v?punpck(l|h)(bw|wd|dq|qdq)", mnemonic)
    if unpack:
        half, suffix = unpack.groups()
        element_bits = {"bw": 8, "wd": 16, "dq": 32, "qdq": 64}[suffix]
        element_mask = (1 << element_bits) - 1
        result = 0
        for lane_base in range(0, vector_bits, 128):
            source_half_offset = lane_base + (64 if half == "h" else 0)
            for element in range(64 // element_bits):
                lane_offset = lane_base + element * element_bits * 2
                left_lane = (a >> (source_half_offset + element * element_bits)) & element_mask
                right_lane = (b >> (source_half_offset + element * element_bits)) & element_mask
                result |= left_lane << lane_offset
                result |= right_lane << (lane_offset + element_bits)
        return _format_int(result & mask)
    pack = re.fullmatch(r"v?pack(sswb|ssdw|uswb|usdw)", mnemonic)
    if pack:
        operation = pack.group(1)
        input_bits, output_bits = (32, 16) if operation.endswith("sdw") else (16, 8)
        input_mask = (1 << input_bits) - 1
        output_mask = (1 << output_bits) - 1
        signed_output = operation.startswith("ss")
        low = -(1 << (output_bits - 1)) if signed_output else 0
        high = (1 << (output_bits - 1)) - 1 if signed_output else output_mask
        result = 0
        for lane_base in range(0, vector_bits, 128):
            output_index = 0
            for source in (a, b):
                for element in range(128 // input_bits):
                    shift = lane_base + element * input_bits
                    value = (source >> shift) & input_mask
                    if value & (1 << (input_bits - 1)):
                        value -= 1 << input_bits
                    packed = min(high, max(low, value)) & output_mask
                    result |= packed << (lane_base + output_index * output_bits)
                    output_index += 1
        return _format_int(result & mask)
    if re.fullmatch(r"v?pmaddwd", mnemonic):
        result = 0
        for offset in range(0, vector_bits, 32):
            total = 0
            for element_offset in (0, 16):
                left_lane = (a >> (offset + element_offset)) & 0xFFFF
                right_lane = (b >> (offset + element_offset)) & 0xFFFF
                if left_lane & 0x8000:
                    left_lane -= 1 << 16
                if right_lane & 0x8000:
                    right_lane -= 1 << 16
                total += left_lane * right_lane
            result |= (total & 0xFFFFFFFF) << offset
        return _format_int(result & mask)
    if re.fullmatch(r"v?pmaddubsw", mnemonic):
        result = 0
        for offset in range(0, vector_bits, 16):
            total = 0
            for element_offset in (0, 8):
                left_lane = (a >> (offset + element_offset)) & 0xFF
                right_lane = (b >> (offset + element_offset)) & 0xFF
                if right_lane & 0x80:
                    right_lane -= 1 << 8
                total += left_lane * right_lane
            result |= (min(0x7FFF, max(-0x8000, total)) & 0xFFFF) << offset
        return _format_int(result & mask)
    if re.fullmatch(r"v?psadbw", mnemonic):
        result = 0
        for offset in range(0, vector_bits, 64):
            total = sum(
                abs(((a >> (offset + byte * 8)) & 0xFF)
                    - ((b >> (offset + byte * 8)) & 0xFF))
                for byte in range(8)
            )
            result |= total << offset
        return _format_int(result & mask)
    horizontal = re.fullmatch(r"v?ph(add|sub)(sw|w|d)", mnemonic)
    if horizontal:
        operation, suffix = horizontal.groups()
        lane_bits = 32 if suffix == "d" else 16
        signed_saturating = suffix == "sw"
        lane_mask = (1 << lane_bits) - 1
        result = 0
        # PH* operates on pairs independently inside each 128-bit lane;
        # each source contributes its results before the next source.
        for lane_base in range(0, vector_bits, 128):
            output_offset = lane_base
            for source in (a, b):
                for pair in range(0, 128, lane_bits * 2):
                    x = (source >> (lane_base + pair)) & lane_mask
                    y = (source >> (lane_base + pair + lane_bits)) & lane_mask
                    if signed_saturating:
                        sign = 1 << (lane_bits - 1)
                        x = x - (1 << lane_bits) if x & sign else x
                        y = y - (1 << lane_bits) if y & sign else y
                    value = x + y if operation == "add" else x - y
                    if signed_saturating:
                        value = min((1 << (lane_bits - 1)) - 1,
                                    max(-(1 << (lane_bits - 1)), value))
                    result |= (value & lane_mask) << output_offset
                    output_offset += lane_bits
        return _format_int(result & mask)
    sign_operation = re.fullmatch(r"v?psign([bwd])", mnemonic)
    if sign_operation:
        lane_bits = {"b": 8, "w": 16, "d": 32}[sign_operation.group(1)]
        lane_mask = (1 << lane_bits) - 1
        result = 0
        for offset in range(0, vector_bits, lane_bits):
            value = (a >> offset) & lane_mask
            control = (b >> offset) & lane_mask
            sign_bit = 1 << (lane_bits - 1)
            signed_control = control - (1 << lane_bits) if control & sign_bit else control
            if signed_control < 0:
                value = -value
            elif signed_control == 0:
                value = 0
            result |= (value & lane_mask) << offset
        return _format_int(result & mask)
    absolute = re.fullmatch(r"v?pabs([bwd])", mnemonic)
    if absolute:
        lane_bits = {"b": 8, "w": 16, "d": 32}[absolute.group(1)]
        lane_mask = (1 << lane_bits) - 1
        sign_bit = 1 << (lane_bits - 1)
        result = 0
        for offset in range(0, vector_bits, lane_bits):
            value = (a >> offset) & lane_mask
            signed_value = value - (1 << lane_bits) if value & sign_bit else value
            result |= (abs(signed_value) & lane_mask) << offset
        return _format_int(result & mask)
    multiply = re.fullmatch(r"v?pmul(lw|hw|huw|ld|dq|udq)", mnemonic)
    if multiply:
        operation = multiply.group(1)
        result = 0
        if operation in {"dq", "udq"}:
            # PMULDQ/PMULUDQ multiply even-numbered dword lanes and pack
            # each full result into the corresponding qword lane.
            signed = operation == "dq"
            for lane_index, output_offset in enumerate(range(0, vector_bits, 64)):
                input_offset = lane_index * 64
                left_lane = (a >> input_offset) & 0xFFFFFFFF
                right_lane = (b >> input_offset) & 0xFFFFFFFF
                if signed:
                    if left_lane & 0x80000000:
                        left_lane -= 1 << 32
                    if right_lane & 0x80000000:
                        right_lane -= 1 << 32
                result |= ((left_lane * right_lane) & ((1 << 64) - 1)) << output_offset
            return _format_int(result & mask)
        lane_bits = 32 if operation == "ld" else 16
        lane_mask = (1 << lane_bits) - 1
        signed = operation in {"lw", "hw"}
        high = operation in {"hw", "huw"}
        for offset in range(0, vector_bits, lane_bits):
            left_lane = (a >> offset) & lane_mask
            right_lane = (b >> offset) & lane_mask
            if signed:
                sign_bit = 1 << (lane_bits - 1)
                if left_lane & sign_bit:
                    left_lane -= 1 << lane_bits
                if right_lane & sign_bit:
                    right_lane -= 1 << lane_bits
            product = left_lane * right_lane
            value = (product >> lane_bits) & lane_mask if high else product & lane_mask
            result |= value << offset
        return _format_int(result & mask)
    match = re.fullmatch(r"v?p(add|sub)([bwdq])", mnemonic)
    if match:
        operation, lane = match.groups()
        lane_bits = {"b": 8, "w": 16, "d": 32, "q": 64}[lane]
        lane_mask = (1 << lane_bits) - 1
        result = 0
        for shift in range(0, vector_bits, lane_bits):
            av, bv = (a >> shift) & lane_mask, (b >> shift) & lane_mask
            lane_value = av + bv if operation == "add" else av - bv
            result |= (lane_value & lane_mask) << shift
        return _format_int(result)
    shift = re.fullmatch(r"v?ps(ll|rl|ra)([wdq])", mnemonic)
    if shift:
        direction, lane = shift.groups()
        lane_bits = {"w": 16, "d": 32, "q": 64}[lane]
        count = b & 0xFF
        lane_mask = (1 << lane_bits) - 1
        result = 0
        for offset in range(0, vector_bits, lane_bits):
            lane_value = (a >> offset) & lane_mask
            if count >= lane_bits:
                shifted = lane_mask if direction == "ra" and lane_value & (1 << (lane_bits - 1)) else 0
            elif direction == "ll":
                shifted = (lane_value << count) & lane_mask
            elif direction == "rl":
                shifted = lane_value >> count
            else:
                signed_lane = lane_value - (1 << lane_bits) if lane_value & (1 << (lane_bits - 1)) else lane_value
                shifted = (signed_lane >> count) & lane_mask
            result |= shifted << offset
        return _format_int(result)
    byte_shift = re.fullmatch(r"v?ps(ll|rl)dq", mnemonic)
    if byte_shift:
        # PSLLDQ/PSRLDQ shift each 128-bit lane independently, including
        # when VEX/EVEX encodings widen the destination vector.
        direction = byte_shift.group(1)
        count = b & 0xFF
        result = 0
        for offset in range(0, vector_bits, 128):
            lane = (a >> offset) & ((1 << 128) - 1)
            if count >= 16:
                shifted = 0
            elif direction == "ll":
                shifted = (lane << (count * 8)) & ((1 << 128) - 1)
            else:
                shifted = lane >> (count * 8)
            result |= shifted << offset
        return _format_int(result)
    saturating = re.fullmatch(r"v?p(add|sub)(us|s)(b|w)", mnemonic)
    if saturating:
        operation, signedness, lane = saturating.groups()
        lane_bits = {"b": 8, "w": 16}[lane]
        lane_mask = (1 << lane_bits) - 1
        signed = signedness == "s"
        low = -(1 << (lane_bits - 1)) if signed else 0
        high = (1 << (lane_bits - 1)) - 1 if signed else lane_mask
        result = 0
        for offset in range(0, vector_bits, lane_bits):
            av, bv = (a >> offset) & lane_mask, (b >> offset) & lane_mask
            if signed:
                sign_bit = 1 << (lane_bits - 1)
                av = av - (1 << lane_bits) if av & sign_bit else av
                bv = bv - (1 << lane_bits) if bv & sign_bit else bv
            value = av + bv if operation == "add" else av - bv
            result |= (min(high, max(low, value)) & lane_mask) << offset
        return _format_int(result)
    average = re.fullmatch(r"v?pavg([bw])", mnemonic)
    compare = re.fullmatch(r"v?pcmp(eq|gt)([bwdq])", mnemonic)
    extrema = re.fullmatch(r"v?p(min|max)(u|s)([bwd])", mnemonic)
    if average or compare or extrema:
        if average:
            lane_bits = {"b": 8, "w": 16}[average.group(1)]
        elif compare:
            lane_bits = {"b": 8, "w": 16, "d": 32, "q": 64}[compare.group(2)]
        else:
            lane_bits = {"b": 8, "w": 16, "d": 32}[extrema.group(3)]
        lane_mask = (1 << lane_bits) - 1
        result = 0
        for offset in range(0, vector_bits, lane_bits):
            av, bv = (a >> offset) & lane_mask, (b >> offset) & lane_mask
            if average:
                value = (av + bv + 1) >> 1
            else:
                signed = bool(compare and compare.group(1) == "gt") or bool(extrema and extrema.group(2) == "s")
                if signed:
                    sign_bit = 1 << (lane_bits - 1)
                    av = av - (1 << lane_bits) if av & sign_bit else av
                    bv = bv - (1 << lane_bits) if bv & sign_bit else bv
                if compare:
                    equal = compare.group(1) == "eq"
                    value = lane_mask if (av == bv if equal else av > bv) else 0
                elif extrema.group(1) == "min":
                    value = min(av, bv)
                else:
                    value = max(av, bv)
                value &= lane_mask
            result |= (value & lane_mask) << offset
        return _format_int(result)
    return _UNKNOWN


def _fold_vector_movemask(mnemonic: str, vector: str, vector_bits: int) -> str:
    """Extract vector element sign bits into the scalar mask result."""
    value = _parse_int(vector)
    match = re.fullmatch(r"v?(movmskps|movmskpd|pmovmskb)", mnemonic)
    if value is None or match is None or vector_bits not in {128, 256, 512}:
        return _UNKNOWN
    operation = match.group(1)
    element_bits = {"movmskps": 32, "movmskpd": 64, "pmovmskb": 8}[operation]
    result = 0
    for index in range(vector_bits // element_bits):
        sign = (value >> (index * element_bits + element_bits - 1)) & 1
        result |= sign << index
    return _format_int(result)


def _fold_vector_shuffle(mnemonic: str, first: str, second: str, control: str,
                         vector_bits: int) -> str:
    """Fold immediate and byte-controlled shuffles within 128-bit lanes."""
    a, b = _parse_int(first), _parse_int(second)
    immediate = _parse_int(control)
    if a is None or b is None or vector_bits not in {128, 256, 512}:
        return _UNKNOWN
    result = 0
    if re.fullmatch(r"v?palignr", mnemonic):
        if immediate is None:
            return _UNKNOWN
        lane_mask = (1 << 128) - 1
        byte_count = immediate & 0xFF
        for lane_base in range(0, vector_bits, 128):
            high = (a >> lane_base) & lane_mask
            low = (b >> lane_base) & lane_mask
            joined = (high << 128) | low
            aligned = 0 if byte_count >= 32 else (joined >> (byte_count * 8)) & lane_mask
            result |= aligned << lane_base
        return _format_int(result)
    blend = re.fullmatch(r"v?(blendps|blendpd|pblendw|pblendd)", mnemonic)
    if blend:
        immediate = _parse_int(control)
        if immediate is None:
            return _UNKNOWN
        operation = blend.group(1)
        lane_bits = {"blendps": 32, "blendpd": 64, "pblendw": 16, "pblendd": 32}[operation]
        lane_mask = (1 << lane_bits) - 1
        for lane_index, offset in enumerate(range(0, vector_bits, lane_bits)):
            selector_bit = lane_index % 8 if operation == "pblendw" else lane_index
            source = b if (immediate >> selector_bit) & 1 else a
            result |= ((source >> offset) & lane_mask) << offset
        return _format_int(result)
    if re.fullmatch(r"v?pshufb", mnemonic):
        for lane_base in range(0, vector_bits, 128):
            for index in range(16):
                selector = (b >> (lane_base + index * 8)) & 0xFF
                value = 0 if selector & 0x80 else (a >> (lane_base + (selector & 0x0F) * 8)) & 0xFF
                result |= value << (lane_base + index * 8)
        return _format_int(result)
    if immediate is None:
        return _UNKNOWN
    if re.fullmatch(r"v?pshufd", mnemonic):
        for lane_base in range(0, vector_bits, 128):
            for index in range(4):
                selector = (immediate >> (2 * index)) & 3
                value = (a >> (lane_base + selector * 32)) & 0xFFFFFFFF
                result |= value << (lane_base + index * 32)
        return _format_int(result)
    if re.fullmatch(r"v?pshuf[hl]w", mnemonic):
        high = "pshufhw" in mnemonic
        for lane_base in range(0, vector_bits, 128):
            original = (a >> lane_base) & ((1 << 128) - 1)
            result |= original << lane_base
            start = 4 if high else 0
            for index in range(4):
                selector = (immediate >> (2 * index)) & 3
                value = (a >> (lane_base + (start + selector) * 16)) & 0xFFFF
                result &= ~(0xFFFF << (lane_base + (start + index) * 16))
                result |= value << (lane_base + (start + index) * 16)
        return _format_int(result)
    if re.fullmatch(r"v?shufps", mnemonic):
        for lane_base in range(0, vector_bits, 128):
            for index in range(4):
                source = a if index < 2 else b
                selector = (immediate >> (2 * (index if index < 2 else index))) & 3
                value = (source >> (lane_base + selector * 32)) & 0xFFFFFFFF
                result |= value << (lane_base + index * 32)
        return _format_int(result)
    if re.fullmatch(r"v?shufpd", mnemonic):
        for lane_index, lane_base in enumerate(range(0, vector_bits, 128)):
            for index in range(2):
                selector = (immediate >> (lane_index * 2 + index)) & 1
                source = a if index == 0 else b
                value = (source >> (lane_base + selector * 64)) & ((1 << 64) - 1)
                result |= value << (lane_base + index * 64)
        return _format_int(result)
    return _UNKNOWN


def _fold_vector_blend_variable(first: str, second: str, control: str,
                                vector_bits: int) -> str:
    """Blend byte lanes by the high bit of each corresponding mask byte."""
    a, b, mask = _parse_int(first), _parse_int(second), _parse_int(control)
    if None in (a, b, mask) or vector_bits not in {128, 256}:
        return _UNKNOWN
    result = 0
    for offset in range(0, vector_bits, 8):
        source = b if (mask >> (offset + 7)) & 1 else a
        result |= ((source >> offset) & 0xFF) << offset
    return _format_int(result)


def _fold_packed_float(mnemonic: str, left: str, right: str, vector_bits: int) -> str:
    """Fold packed SSE/AVX arithmetic only when every lane is exactly representable."""
    operation_match = re.fullmatch(r"v?(add|sub|mul|div)(ps|pd)", mnemonic)
    a, b = _parse_int(left), _parse_int(right)
    if operation_match is None or a is None or b is None or vector_bits not in {128, 256, 512}:
        return _UNKNOWN
    operation, precision = operation_match.groups()
    lane_bits = 64 if precision == "pd" else 32
    byte_count = lane_bits // 8
    fmt = "<d" if lane_bits == 64 else "<f"
    lane_mask = (1 << lane_bits) - 1
    minimum_normal = 2.0 ** (-1022 if lane_bits == 64 else -126)
    result = 0
    for offset in range(0, vector_bits, lane_bits):
        left_bits = (a >> offset) & lane_mask
        right_bits = (b >> offset) & lane_mask
        x = struct.unpack(fmt, left_bits.to_bytes(byte_count, "little"))[0]
        y = struct.unpack(fmt, right_bits.to_bytes(byte_count, "little"))[0]
        if (not math.isfinite(x) or not math.isfinite(y)
                or (x != 0.0 and abs(x) < minimum_normal)
                or (y != 0.0 and abs(y) < minimum_normal)):
            return _UNKNOWN
        exact_x, exact_y = Fraction.from_float(x), Fraction.from_float(y)
        if operation == "add":
            exact = exact_x + exact_y
        elif operation == "sub":
            exact = exact_x - exact_y
        elif operation == "mul":
            exact = exact_x * exact_y
        elif exact_y:
            exact = exact_x / exact_y
        else:
            return _UNKNOWN
        # Exact zero can have a sign determined by MXCSR rounding; leave it
        # unknown rather than inventing that sign or exception behavior.
        if not exact:
            return _UNKNOWN
        try:
            packed = struct.pack(fmt, float(exact))
            folded = struct.unpack(fmt, packed)[0]
        except (OverflowError, struct.error):
            return _UNKNOWN
        if (not math.isfinite(folded)
                or (folded != 0.0 and abs(folded) < minimum_normal)
                or Fraction.from_float(folded) != exact):
            return _UNKNOWN
        result |= int.from_bytes(packed, "little") << offset
    return _format_int(result)


def _fold_packed_float_compare(mnemonic: str, left: str, right: str,
                               predicate: str, vector_bits: int) -> str:
    """Fold the eight SSE ordered/unordered packed comparison predicates."""
    comparison = re.fullmatch(r"v?cmpp([sd])", mnemonic)
    a, b, selector = _parse_int(left), _parse_int(right), _parse_int(predicate)
    predicate_limit = 32 if mnemonic.startswith("v") else 8
    if (comparison is None or a is None or b is None or selector is None
            or selector not in range(predicate_limit) or vector_bits not in {128, 256}):
        return _UNKNOWN
    lane_bits = 64 if comparison.group(1) == "d" else 32
    lane_mask = (1 << lane_bits) - 1
    fmt = "<d" if lane_bits == 64 else "<f"
    byte_count = lane_bits // 8
    result = 0
    for offset in range(0, vector_bits, lane_bits):
        x = struct.unpack(fmt, ((a >> offset) & lane_mask).to_bytes(byte_count, "little"))[0]
        y = struct.unpack(fmt, ((b >> offset) & lane_mask).to_bytes(byte_count, "little"))[0]
        unordered = math.isnan(x) or math.isnan(y)
        ordered = not unordered
        predicates = (
            ordered and x == y,
            ordered and x < y,
            ordered and x <= y,
            unordered,
            unordered or x != y,
            unordered or x >= y,
            unordered or x > y,
            ordered,
            unordered or x == y,
            unordered or x < y,
            unordered or x <= y,
            False,
            ordered and x != y,
            ordered and x >= y,
            ordered and x > y,
            True,
            ordered and x == y,
            ordered and x < y,
            ordered and x <= y,
            unordered,
            unordered or x != y,
            unordered or x >= y,
            unordered or x > y,
            ordered,
            unordered or x == y,
            unordered or x < y,
            unordered or x <= y,
            False,
            ordered and x != y,
            ordered and x >= y,
            ordered and x > y,
            True,
        )
        if predicates[selector]:
            result |= lane_mask << offset
    return _format_int(result)


def _fold_packed_float_minmax(mnemonic: str, left: str, right: str,
                              vector_bits: int) -> str:
    """Fold packed MIN/MAX while preserving source bits on ties and NaNs."""
    operation_match = re.fullmatch(r"v?(min|max)(ps|pd)", mnemonic)
    a, b = _parse_int(left), _parse_int(right)
    if operation_match is None or a is None or b is None or vector_bits not in {128, 256, 512}:
        return _UNKNOWN
    operation, precision = operation_match.groups()
    lane_bits = 64 if precision == "pd" else 32
    lane_mask = (1 << lane_bits) - 1
    byte_count = lane_bits // 8
    fmt = "<d" if lane_bits == 64 else "<f"
    result = 0
    for offset in range(0, vector_bits, lane_bits):
        x_bits, y_bits = (a >> offset) & lane_mask, (b >> offset) & lane_mask
        x = struct.unpack(fmt, x_bits.to_bytes(byte_count, "little"))[0]
        y = struct.unpack(fmt, y_bits.to_bytes(byte_count, "little"))[0]
        if math.isnan(x) or math.isnan(y) or x == y:
            chosen = y_bits
        elif (x < y if operation == "min" else x > y):
            chosen = x_bits
        else:
            chosen = y_bits
        result |= chosen << offset
    return _format_int(result)


def _fold_packed_float_sqrt(mnemonic: str, source: str, vector_bits: int) -> str:
    """Fold only nonnegative packed square roots exactly representable per lane."""
    operation_match = re.fullmatch(r"v?sqrt(ps|pd)", mnemonic)
    value = _parse_int(source)
    if operation_match is None or value is None or vector_bits not in {128, 256, 512}:
        return _UNKNOWN
    lane_bits = 64 if operation_match.group(1) == "pd" else 32
    lane_mask = (1 << lane_bits) - 1
    byte_count = lane_bits // 8
    fmt = "<d" if lane_bits == 64 else "<f"
    minimum_normal = 2.0 ** (-1022 if lane_bits == 64 else -126)
    result = 0
    for offset in range(0, vector_bits, lane_bits):
        lane_bits_value = (value >> offset) & lane_mask
        lane = struct.unpack(fmt, lane_bits_value.to_bytes(byte_count, "little"))[0]
        if lane == 0.0:
            result |= lane_bits_value << offset
            continue
        if not math.isfinite(lane) or lane < 0.0 or abs(lane) < minimum_normal:
            return _UNKNOWN
        exact = Fraction.from_float(lane)
        numerator_root = math.isqrt(exact.numerator)
        denominator_root = math.isqrt(exact.denominator)
        if (numerator_root * numerator_root != exact.numerator
                or denominator_root * denominator_root != exact.denominator):
            return _UNKNOWN
        root = Fraction(numerator_root, denominator_root)
        try:
            packed = struct.pack(fmt, float(root))
            folded = struct.unpack(fmt, packed)[0]
        except (OverflowError, struct.error):
            return _UNKNOWN
        if (not math.isfinite(folded) or (folded and abs(folded) < minimum_normal)
                or Fraction.from_float(folded) != root):
            return _UNKNOWN
        result |= int.from_bytes(packed, "little") << offset
    return _format_int(result)


def _fold_packed_float_int_convert(mnemonic: str, source: str, vector_bits: int) -> str:
    """Fold packed signed dword/float conversions under known rounding rules."""
    operation = re.fullmatch(r"v?(cvtdq2ps|cvttps2dq|cvtps2dq)", mnemonic)
    value = _parse_int(source)
    if operation is None or value is None or vector_bits not in {128, 256, 512}:
        return _UNKNOWN
    kind = operation.group(1)
    result = 0
    for offset in range(0, vector_bits, 32):
        lane = (value >> offset) & 0xFFFFFFFF
        if kind == "cvtdq2ps":
            signed = lane - (1 << 32) if lane & 0x80000000 else lane
            try:
                packed = struct.pack("<f", float(signed))
                converted = struct.unpack("<f", packed)[0]
            except (OverflowError, struct.error):
                return _UNKNOWN
            if Fraction.from_float(converted) != signed:
                return _UNKNOWN
            result |= int.from_bytes(packed, "little") << offset
            continue
        source_float = struct.unpack("<f", lane.to_bytes(4, "little"))[0]
        if (not math.isfinite(source_float)
                or (source_float != 0.0 and abs(source_float) < 2.0 ** -126)):
            return _UNKNOWN
        if kind == "cvttps2dq":
            integer = math.trunc(source_float)
        elif source_float.is_integer():
            integer = int(source_float)
        else:
            # CVTPS2DQ depends on MXCSR for non-integral values.
            return _UNKNOWN
        if not -(1 << 31) <= integer < (1 << 31):
            return _UNKNOWN
        result |= (integer & 0xFFFFFFFF) << offset
    return _format_int(result)


def _fold_packed_float_precision_convert(mnemonic: str, source: str,
                                         source_bits: int, destination_bits: int) -> str:
    """Fold packed single/double conversions when each result is exact."""
    operation = re.fullmatch(r"v?(cvtps2pd|cvtpd2ps)", mnemonic)
    value = _parse_int(source)
    if (operation is None or value is None
            or source_bits not in {128, 256, 512}
            or destination_bits not in {128, 256, 512}):
        return _UNKNOWN
    widen = operation.group(1) == "cvtps2pd"
    source_lane_bits, destination_lane_bits = (32, 64) if widen else (64, 32)
    lane_count = destination_bits // 64 if widen else source_bits // 64
    if lane_count > destination_bits // destination_lane_bits:
        return _UNKNOWN
    source_mask = (1 << source_lane_bits) - 1
    source_fmt = "<f" if source_lane_bits == 32 else "<d"
    destination_fmt = "<d" if destination_lane_bits == 64 else "<f"
    result = 0
    for index in range(lane_count):
        source_lane = (value >> (index * source_lane_bits)) & source_mask
        source_value = struct.unpack(
            source_fmt, source_lane.to_bytes(source_lane_bits // 8, "little")
        )[0]
        if math.isnan(source_value):
            return _UNKNOWN
        try:
            packed = struct.pack(destination_fmt, source_value)
            converted = struct.unpack(destination_fmt, packed)[0]
        except (OverflowError, struct.error):
            return _UNKNOWN
        exact = (
            math.isinf(source_value) and math.isinf(converted)
            and math.copysign(1.0, source_value) == math.copysign(1.0, converted)
        ) or (
            math.isfinite(source_value) and math.isfinite(converted)
            and Fraction.from_float(source_value) == Fraction.from_float(converted)
            and (source_value != 0.0
                 or math.copysign(1.0, source_value) == math.copysign(1.0, converted))
        )
        if not exact:
            return _UNKNOWN
        result |= int.from_bytes(packed, "little") << (index * destination_lane_bits)
    return _format_int(result)


def _result_condition_flags(result: int, width_bits: int) -> tuple[bool, bool, bool]:
    mask = (1 << width_bits) - 1
    value = result & mask
    zero = value == 0
    sign = bool(value & (1 << (width_bits - 1)))
    parity = (value & 0xFF).bit_count() % 2 == 0
    return zero, sign, parity


def _arithmetic_condition_flags(
    operation: str, left: int, right: int, width_bits: int, carry_in: int = 0
) -> tuple[int, bool, bool, bool, bool, bool, bool]:
    """Return result and x86 CF/PF/AF/ZF/SF/OF for known integer arithmetic."""
    mask = (1 << width_bits) - 1
    sign_bit = 1 << (width_bits - 1)
    a, b = left & mask, right & mask
    if operation in {"sub", "sbb", "cmp"}:
        subtrahend = b + carry_in
        result = (a - subtrahend) & mask
        carry = a < subtrahend
        signed_a = a - (1 << width_bits) if a & sign_bit else a
        signed_b = b - (1 << width_bits) if b & sign_bit else b
        signed_result = signed_a - signed_b - carry_in
        overflow = not (-(1 << (width_bits - 1)) <= signed_result < (1 << (width_bits - 1)))
        auxiliary = (a & 0xF) < ((b & 0xF) + carry_in)
    else:
        total = a + b + carry_in
        result = total & mask
        carry = total > mask
        signed_a = a - (1 << width_bits) if a & sign_bit else a
        signed_b = b - (1 << width_bits) if b & sign_bit else b
        signed_result = signed_a + signed_b + carry_in
        overflow = not (-(1 << (width_bits - 1)) <= signed_result < (1 << (width_bits - 1)))
        auxiliary = (a & 0xF) + (b & 0xF) + carry_in > 0xF
    zero, sign, parity = _result_condition_flags(result, width_bits)
    return result, carry, parity, auxiliary, zero, sign, overflow


@dataclass
class AbstractState:
    regs: dict[str, str] = field(default_factory=dict)
    stack: dict[str, str] = field(default_factory=dict)
    memory: dict[str, str] = field(default_factory=dict)
    flags: tuple[str, str, str] | None = None  # operation, left expression, right expression
    reg_types: dict[str, str] = field(default_factory=dict)
    stack_types: dict[str, str] = field(default_factory=dict)
    stack_widths: dict[str, int] = field(default_factory=dict)
    memory_widths: dict[str, int] = field(default_factory=dict)
    carry: bool | None = None
    zero: bool | None = None
    sign: bool | None = None
    overflow: bool | None = None
    parity: bool | None = None
    auxiliary: bool | None = None
    direction: bool | None = None
    pending_pushes: list[str] = field(default_factory=list)
    reg_slices: dict[str, str] = field(default_factory=dict)
    x87_stack: list[str] | None = field(default_factory=list)
    memory_types: dict[str, str] = field(default_factory=dict)

    def clone(self) -> "AbstractState":
        return AbstractState(dict(self.regs), dict(self.stack), dict(self.memory), self.flags,
                             dict(self.reg_types), dict(self.stack_types), dict(self.stack_widths),
                             dict(self.memory_widths), self.carry, self.zero, self.sign,
                             self.overflow, self.parity, self.auxiliary, self.direction,
                             list(self.pending_pushes), dict(self.reg_slices),
                             list(self.x87_stack) if self.x87_stack is not None else None,
                             dict(self.memory_types))

    @staticmethod
    def merge(left: "AbstractState", right: "AbstractState") -> "AbstractState":
        # A value survives a join only if every incoming path agrees.
        regs = {k: v for k, v in left.regs.items() if right.regs.get(k) == v and v != _UNKNOWN}
        stack = {
            k: v for k, v in left.stack.items()
            if right.stack.get(k) == v and v != _UNKNOWN
            and left.stack_widths.get(k) == right.stack_widths.get(k)
        }
        stack_widths = {k: left.stack_widths[k] for k in stack if k in left.stack_widths}
        memory = {k: v for k, v in left.memory.items() if right.memory.get(k) == v and v != _UNKNOWN}
        memory = {
            k: v for k, v in memory.items()
            if left.memory_widths.get(k) == right.memory_widths.get(k)
        }
        memory_widths = {k: left.memory_widths[k] for k in memory if k in left.memory_widths}
        flags = left.flags if left.flags == right.flags else None
        carry = left.carry if left.carry == right.carry else None
        zero = left.zero if left.zero == right.zero else None
        sign = left.sign if left.sign == right.sign else None
        overflow = left.overflow if left.overflow == right.overflow else None
        parity = left.parity if left.parity == right.parity else None
        auxiliary = left.auxiliary if left.auxiliary == right.auxiliary else None
        direction = left.direction if left.direction == right.direction else None
        reg_slices = {key: value for key, value in left.reg_slices.items()
                      if right.reg_slices.get(key) == value and value != _UNKNOWN}
        x87_stack = left.x87_stack if left.x87_stack == right.x87_stack else None
        reg_types = {
            key: (value if right.reg_types.get(key) == value else _TYPE_UNKNOWN)
            for key, value in left.reg_types.items()
            if key in regs
        }
        stack_types = {
            key: (value if right.stack_types.get(key) == value else _TYPE_UNKNOWN)
            for key, value in left.stack_types.items()
            if key in stack
        }
        memory_types = {
            key: (left.memory_types.get(key, _TYPE_UNKNOWN)
                  if right.memory_types.get(key, _TYPE_UNKNOWN) == left.memory_types.get(key, _TYPE_UNKNOWN)
                  else _TYPE_UNKNOWN)
            for key in memory
        }
        return AbstractState(regs, stack, memory, flags, reg_types, stack_types,
                             stack_widths, memory_widths, carry, zero, sign, overflow,
                             parity, auxiliary, direction,
                             list(left.pending_pushes) if left.pending_pushes == right.pending_pushes else [],
                             reg_slices, list(x87_stack) if x87_stack is not None else None,
                             memory_types)

    def to_dict(self) -> dict[str, Any]:
        return {"registers": dict(sorted(self.regs.items())),
                "stack_slots": dict(sorted(self.stack.items())),
                "stack_slot_widths": dict(sorted(self.stack_widths.items())),
                "memory": dict(sorted(self.memory.items())),
                "memory_widths": dict(sorted(self.memory_widths.items())),
                "memory_types": dict(sorted(self.memory_types.items())),
                "register_types": dict(sorted(self.reg_types.items())),
                "stack_slot_types": dict(sorted(self.stack_types.items())),
                "last_compare": list(self.flags) if self.flags else None,
                "carry_flag": self.carry, "zero_flag": self.zero,
                "sign_flag": self.sign, "overflow_flag": self.overflow,
                "parity_flag": self.parity, "auxiliary_flag": self.auxiliary,
                "direction_flag": self.direction,
                "pending_push_arguments": list(self.pending_pushes),
                "register_slices": dict(sorted(self.reg_slices.items())),
                "x87_stack": list(self.x87_stack) if self.x87_stack is not None else None}


def _clear_register_slices(state: AbstractState, family: str) -> None:
    state.reg_slices = {alias: value for alias, value in state.reg_slices.items()
                        if _canon_reg(alias) != family}


def _x87_index(operand: str) -> int | None:
    match = re.fullmatch(r"\s*st(?:\s*\(\s*([0-7])\s*\))?\s*", operand, re.I)
    if not match:
        return None
    return int(match.group(1) or 0)


def _parse_x87_number(value: str) -> Decimal | None:
    text = value.strip()
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", text, re.I):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _fold_exact_x87_integer_arithmetic(left: str, right: str, operation: str) -> str | None:
    """Fold integer-valued x87 operations only when the result is exact and in range."""
    lhs = _parse_x87_number(left)
    rhs = _parse_x87_number(right)
    if lhs is None or rhs is None or lhs != lhs.to_integral_value() or rhs != rhs.to_integral_value():
        return None
    left_int, right_int = int(lhs), int(rhs)
    if operation == "+":
        result = left_int + right_int
    elif operation == "-":
        result = left_int - right_int
    elif operation == "*":
        result = left_int * right_int
    elif operation == "/":
        if right_int == 0 or left_int % right_int != 0:
            return None
        result = left_int // right_int
    else:
        return None
    # Every signed 64-bit integer is exactly representable in x87 extended
    # precision. Larger results may round according to the control word, which
    # this analysis does not track.
    if abs(result) > (1 << 63):
        return None
    return f"{result}.0"


def _canonical_memory_range(key: str) -> tuple[str, int] | None:
    absolute = re.fullmatch(r"va:(0x[0-9a-f]+|\d+)", key, re.I)
    if absolute:
        return "va", _parse_int(absolute.group(1)) or 0
    symbolic = re.fullmatch(r"(symbol:[^+\-]+)([+-]\d+)?", key, re.I)
    if symbolic:
        return symbolic.group(1), int(symbolic.group(2) or "0", 10)
    return None


class FunctionDataFlow:
    """Bounded block-level value propagation for one candidate function."""

    def __init__(self, bits: int, strings: dict[int, str], imports: dict[int, str],
                 function_names: dict[int, str], function_arg_counts: dict[int, int] | None = None,
                 function_return_types: dict[int, str] | None = None,
                 known_code_addresses: set[int] | None = None,
                 max_steps_per_function: int = 20_000, max_visits_per_block: int = 8,
                 static_pointer_slots: dict[int, int] | None = None):
        self.bits = bits
        self.strings = strings
        self.imports = imports
        self.function_names = function_names
        self.known_code_addresses = known_code_addresses or set(function_names)
        self.static_pointer_slots = static_pointer_slots or {}
        self.function_arg_counts = function_arg_counts or {}
        self.function_return_types = function_return_types or {}
        self.max_steps_per_function = max_steps_per_function
        self.max_visits_per_block = max_visits_per_block
        self.steps = 0
        self.warnings: list[str] = []

    @staticmethod
    def _pointer_expression(value: str) -> tuple[str, int] | None:
        value = value.strip()
        static = re.fullmatch(r"static_ptr\((0x[0-9a-f]+|\d+)\)", value, re.I)
        cast = re.fullmatch(r"\(void \*\)(0x[0-9a-f]+|\d+)", value, re.I)
        number = _parse_int(value)
        if static or cast or number is not None:
            return f"va:{_parse_int((static or cast).group(1)) if static or cast else number:#x}", 0
        canonical_symbol = re.fullmatch(r"(symbol:(?:arg\d+|v_[a-z0-9]+))([+-]\d+)?", value, re.I)
        if canonical_symbol:
            return canonical_symbol.group(1), int(canonical_symbol.group(2) or "0", 10)
        indexed = re.fullmatch(r"(idxexpr\(.+\))([+-]\d+)?", value, re.I)
        if indexed:
            return indexed.group(1), int(indexed.group(2) or "0", 10)
        arithmetic = re.fullmatch(r"\((.+)\s*([+-])\s*(0x[0-9a-f]+|\d+)\)", value, re.I)
        if arithmetic:
            base = FunctionDataFlow._pointer_expression(arithmetic.group(1))
            amount = _parse_int(arithmetic.group(3))
            if base is not None and amount is not None:
                return base[0], base[1] + (amount if arithmetic.group(2) == "+" else -amount)
        if re.fullmatch(r"(?:arg\d+|v_[a-z0-9]+)", value, re.I):
            return f"symbol:{value}", 0
        return None

    @staticmethod
    def _address_key_with_offset(root: str, offset: int) -> str:
        if root.startswith("va:"):
            return f"va:{(int(root[3:], 16) + offset):#x}"
        if root.startswith("idxexpr(") and offset == 0:
            return root
        return f"{root}{offset:+d}"

    def _memory_key(self, inner: str, state: AbstractState) -> str:
        """Canonicalize simple base-plus-constant addresses through known pointers."""
        absolute = _parse_int(inner)
        if absolute is not None:
            return f"va:{absolute:#x}"
        scaled_terms = list(re.finditer(
            r"\b([a-z][a-z0-9]*)\s*\*\s*(2|4|8)\b", inner, re.I
        ))
        if 2 <= len(scaled_terms) <= 4:
            remainder = inner
            for term in reversed(scaled_terms):
                remainder = remainder[:term.start()] + " " + remainder[term.end():]
            unscaled = list(re.finditer(r"\b([a-z][a-z0-9]*)\b", remainder, re.I))
            if len(unscaled) == 1:
                base_token = unscaled[0].group(1)
                base_family = _canon_reg(base_token)
                pointer = self._pointer_expression(state.regs.get(base_family or "", ""))
                if pointer is not None:
                    remainder = remainder[:unscaled[0].start()] + " " + remainder[unscaled[0].end():]
                    displacement = 0
                    numeric_terms = list(re.finditer(r"([+-]?)\s*(0x[0-9a-f]+|\d+)", remainder, re.I))
                    for numeric in numeric_terms:
                        parsed = _parse_int(numeric.group(2))
                        if parsed is not None:
                            displacement += -parsed if numeric.group(1) == "-" else parsed
                    residue = re.sub(r"([+-]?)\s*(0x[0-9a-f]+|\d+)", "", remainder, flags=re.I)
                    if re.sub(r"[+\-\s]", "", residue):
                        return inner
                    symbolic_indices: list[tuple[str, int, int]] = []
                    valid_indices = True
                    for term in scaled_terms:
                        family = _canon_reg(term.group(1))
                        value = state.regs.get(family or "")
                        preceding = inner[:term.start()].rstrip()
                        sign_match = re.search(r"([+-])$", preceding)
                        sign = -1 if sign_match and sign_match.group(1) == "-" else 1
                        scale = int(term.group(2)) * sign
                        number = _parse_int(value) if value is not None else None
                        if number is not None:
                            displacement += number * scale
                            continue
                        expression = self._pointer_expression(value or "")
                        if expression is None or not expression[0].startswith("symbol:"):
                            valid_indices = False
                            break
                        symbolic_indices.append((expression[0], expression[1], scale))
                    if valid_indices:
                        if not symbolic_indices:
                            return self._address_key_with_offset(
                                pointer[0], pointer[1] + displacement
                            )
                        index_key = ",".join(
                            f"{root}{offset:+d}*{scale}"
                            for root, offset, scale in sorted(symbolic_indices)
                        )
                        indexed_root = f"idxexpr({self._address_key_with_offset(pointer[0], pointer[1])}|{index_key})"
                        return self._address_key_with_offset(indexed_root, displacement)
        if len(scaled_terms) == 1:
            scaled = scaled_terms[0]
            remainder = inner[:scaled.start()] + " " + inner[scaled.end():]
            index_family = _canon_reg(scaled.group(1))
            register_terms = [
                (match.group(1), _canon_reg(match.group(1)))
                for match in re.finditer(r"\b([a-z][a-z0-9]*)\b", remainder, re.I)
                if _canon_reg(match.group(1)) is not None
            ]
            if len(register_terms) == 1 and index_family is not None:
                base_token, base_family = register_terms[0]
                pointer = self._pointer_expression(state.regs.get(base_family, ""))
                index_value = state.regs.get(index_family)
                index_number = _parse_int(index_value) if index_value is not None else None
                index_expression = self._pointer_expression(index_value) if index_value is not None else None
                remainder_without_base = re.sub(
                    rf"\b{re.escape(base_token)}\b", " ", remainder, count=1, flags=re.I
                )
                displacement = 0
                for sign, number in re.findall(r"([+-])\s*(0x[0-9a-f]+|\d+)", remainder_without_base, re.I):
                    parsed = _parse_int(number)
                    if parsed is not None:
                        displacement += parsed if sign == "+" else -parsed
                scale = int(scaled.group(2))
                if pointer is not None and index_number is not None:
                    total_offset = pointer[1] + displacement + index_number * scale
                    return self._address_key_with_offset(pointer[0], total_offset)
                if pointer is not None and index_expression is not None:
                    index_root, index_offset = index_expression
                    indexed_root = (
                        f"idxexpr({self._address_key_with_offset(pointer[0], pointer[1])}|"
                        f"{index_root}{index_offset:+d}*{scale})"
                    )
                    return self._address_key_with_offset(indexed_root, displacement)
        expression = re.fullmatch(
            r"([a-z][a-z0-9]*)(?:\s*([+-])\s*(0x[0-9a-f]+|\d+))?", inner, re.I
        )
        if not expression:
            return inner
        register, operator, offset_text = expression.groups()
        register_family = _canon_reg(register)
        register_value = state.regs.get(register_family or "")
        base = self._pointer_expression(register_value) if register_value is not None else None
        offset = _parse_int(offset_text) if offset_text else 0
        if base is None or offset is None:
            return inner
        effective_offset = base[1] + (offset if operator != "-" else -offset)
        return self._address_key_with_offset(base[0], effective_offset)

    def initial_state(self) -> AbstractState:
        # Windows x64 integer/pointer argument registers; these are symbolic parameters,
        # not claims that every argument is actually used by the recovered function.
        s = AbstractState()
        if self.bits == 64:
            for reg, name in (("rcx", "arg1"), ("rdx", "arg2"), ("r8", "arg3"), ("r9", "arg4")):
                s.regs[reg] = name
                s.reg_types[reg] = "uint64_t"
        else:
            for reg, name in (("ecx", "arg1"), ("edx", "arg2")):
                canonical = _canon_reg(reg)
                if canonical:
                    s.regs[canonical] = name
                    s.reg_types[canonical] = "uint32_t"
        return s

    def resolve(self, operand: str, state: AbstractState, ins: Instruction | None = None) -> str:
        value = operand.strip()
        reg = _canon_reg(value)
        if reg:
            register_slice = _register_slice(value, self.bits)
            if register_slice is not None:
                family, width, offset = register_slice
                exact = state.reg_slices.get(value.lower())
                if exact is not None:
                    return exact
                requested_end = offset + width
                for alias, slice_value in state.reg_slices.items():
                    alias_slice = _register_slice(alias, self.bits)
                    number = _parse_int(slice_value)
                    if (alias_slice is None or alias_slice[0] != family or number is None
                            or alias_slice[2] > offset
                            or alias_slice[2] + alias_slice[1] < requested_end):
                        continue
                    return _format_int((number >> (offset - alias_slice[2])) & ((1 << width) - 1))
                full = state.regs.get(family)
                full_number = _parse_int(full or "")
                if full_number is not None:
                    return _format_int((full_number >> offset) & ((1 << width) - 1))
                if full == _UNKNOWN:
                    return _UNKNOWN
                if width == 16:
                    low = state.reg_slices.get({"rax": "al", "rbx": "bl", "rcx": "cl", "rdx": "dl"}.get(family, ""))
                    high = state.reg_slices.get({"rax": "ah", "rbx": "bh", "rcx": "ch", "rdx": "dh"}.get(family, ""))
                    low_number, high_number = _parse_int(low or ""), _parse_int(high or "")
                    if low_number is not None and high_number is not None:
                        return _format_int((high_number & 0xFF) << 8 | (low_number & 0xFF))
                return full if full is not None else value.lower()
            return state.regs.get(reg, value.lower())
        inner = _mem_inner(value)
        inner = _canonical_mem_inner(inner)
        if inner is not None:
            if _is_stack_mem(inner) and inner in state.stack:
                stored_width = state.stack_widths.get(inner)
                requested_width = _operand_width_bytes(value, self.bits)
                if stored_width is None or stored_width == requested_width:
                    return state.stack[inner]
                return _UNKNOWN
            if _is_stack_mem(inner):
                argument = _stack_argument_name(inner)
                if argument:
                    return argument
            memory_key = self._memory_key(inner, state) if not _is_stack_mem(inner) else inner
            if memory_key in state.memory:
                stored_width = state.memory_widths.get(memory_key)
                requested_width = _operand_width_bytes(value, self.bits)
                if stored_width is not None and stored_width != requested_width:
                    return _UNKNOWN
                return state.memory[memory_key]
            if ins and ins.reference_target in self.imports:
                return f"iat({self.imports[ins.reference_target]})"
            if ins and ins.reference_target in self.static_pointer_slots:
                return f"static_ptr(0x{self.static_pointer_slots[ins.reference_target]:X})"
            if ins and "rip" in inner and ins.reference_target is not None:
                if ins.reference_target in self.strings:
                    return f"string_at(0x{ins.reference_target:X})"
                if ins.reference_target in self.imports:
                    return f"iat({self.imports[ins.reference_target]})"
                if ins.reference_target in self.static_pointer_slots:
                    return f"static_ptr(0x{self.static_pointer_slots[ins.reference_target]:X})"
                return f"mem_at(0x{ins.reference_target:X})"
            address_match = re.fullmatch(
                r"([a-z][a-z0-9]*)(?:\s*([+-])\s*(0x[0-9a-f]+|\d+))?", inner, re.I
            )
            if address_match:
                base_reg, operator, offset_text = address_match.groups()
                base_value = state.regs.get(_canon_reg(base_reg) or "", "")
                static_match = re.fullmatch(r"static_ptr\((0x[0-9a-f]+|\d+)\)", base_value, re.I)
                base_address = None
                if static_match:
                    base_address = _parse_int(static_match.group(1))
                else:
                    address_value = re.fullmatch(r"(?:\(void \*\)\s*)?(0x[0-9a-f]+|\d+)", base_value, re.I)
                    base_address = _parse_int(address_value.group(1)) if address_value else _parse_int(base_value)
                offset = _parse_int(offset_text) if offset_text else 0
                if base_address is not None and offset is not None:
                    effective_address = base_address + (offset if operator != "-" else -offset)
                    pointer = self.static_pointer_slots.get(effective_address)
                    if pointer is not None:
                        return f"static_ptr(0x{pointer:X})"
            return f"mem[{inner}]"
        if ins and ins.reference_target is not None and value.lower().startswith("0x"):
            if ins.reference_target in self.strings:
                return f"string_at(0x{ins.reference_target:X})"
        number = _parse_int(value)
        if number is not None:
            return _format_int(number)
        return value

    def _assign(self, dst: str, value: str, state: AbstractState,
                source_operand: str | None = None,
                zero_upper_vector: bool = False) -> tuple[str, str] | None:
        reg = _canon_reg(dst)
        if reg:
            register_slice = _register_slice(dst, self.bits)
            if register_slice is not None:
                family, width, offset = register_slice
                alias = dst.strip().lower()
                mask = (1 << width) - 1
                numeric = _parse_int(value)
                stored = _format_int(numeric & mask) if numeric is not None else value
                if width >= self.bits or (self.bits == 64 and width == 32 and offset == 0):
                    state.reg_slices = {key: val for key, val in state.reg_slices.items()
                                        if _canon_reg(key) != family}
                    state.regs[family] = stored
                else:
                    start, end = offset, offset + width
                    for key in list(state.reg_slices):
                        key_slice = _register_slice(key, self.bits)
                        if key_slice is None or key_slice[0] != family:
                            continue
                        key_start, key_end = key_slice[2], key_slice[2] + key_slice[1]
                        if start < key_end and key_start < end:
                            state.reg_slices.pop(key, None)
                    state.reg_slices[alias] = stored
                    old = _parse_int(state.regs.get(family, ""))
                    if old is not None and numeric is not None:
                        full_mask = (1 << self.bits) - 1
                        field_mask = mask << offset
                        combined = (old & (full_mask ^ field_mask)) | ((numeric & mask) << offset)
                        state.regs[family] = _format_int(combined & full_mask)
                    else:
                        state.regs[family] = _UNKNOWN
                return (reg, stored)
            vector = re.fullmatch(r"([xyz])mm([0-9]|[12][0-9]|3[01])", dst.strip().lower())
            if vector:
                prefix, number = vector.groups()
                if zero_upper_vector:
                    widths = {"x": 128, "y": 256, "z": 512}
                    destination_width = widths[prefix]
                    numeric = _parse_int(value)
                    for alias, alias_width in widths.items():
                        alias_name = f"{alias}mm{number}"
                        if numeric is not None:
                            alias_value = numeric & ((1 << min(alias_width, destination_width)) - 1)
                            alias_stored = _format_int(alias_value)
                        elif value == _UNKNOWN:
                            alias_stored = _UNKNOWN
                        elif alias_width < destination_width:
                            alias_stored = f"extract_low_{alias_width}({value})"
                        elif alias_width > destination_width:
                            alias_stored = f"zero_extend_{destination_width}({value})"
                        else:
                            alias_stored = value
                        state.regs[alias_name] = alias_stored
                        state.reg_types.pop(alias_name, None)
                else:
                    for alias in ("x", "y", "z"):
                        if alias != prefix:
                            state.regs.pop(f"{alias}mm{number}", None)
                            state.reg_types.pop(f"{alias}mm{number}", None)
                    if zero_upper_vector:
                        return (reg, state.regs[reg])
            state.regs[reg] = value
            return (reg, value)
        inner = _mem_inner(dst)
        inner = _canonical_mem_inner(inner)
        if _is_stack_mem(inner):
            assert inner is not None
            address = _stack_address(inner)
            width = _operand_width_bytes(dst, self.bits)
            if source_operand and not re.match(r"\s*(?:byte|word|dword|qword|xmmword|ymmword|zmmword)\s+ptr\b", dst, re.I):
                width = _operand_width_bytes(source_operand, self.bits)
            if address is not None:
                base, start = address
                end = start + width
                for existing in list(state.stack):
                    existing_address = _stack_address(existing)
                    if existing_address is None or existing_address[0] != base:
                        continue
                    existing_start = existing_address[1]
                    existing_end = existing_start + state.stack_widths.get(existing, self.bits // 8)
                    if start < existing_end and existing_start < end:
                        state.stack.pop(existing, None)
                        state.stack_types.pop(existing, None)
                        state.stack_widths.pop(existing, None)
            state.stack[inner] = value
            state.stack_widths[inner] = width
            return (f"stack:{inner}", value)
        if inner is not None:
            memory_key = self._memory_key(inner, state)
            width = _operand_width_bytes(dst, self.bits)
            if source_operand and not re.match(r"\s*(?:byte|word|dword|qword|xmmword|ymmword|zmmword)\s+ptr\b", dst, re.I):
                width = _operand_width_bytes(source_operand, self.bits)
            write_range = _canonical_memory_range(memory_key)
            for existing_key in list(state.memory):
                existing_range = _canonical_memory_range(existing_key)
                existing_width = state.memory_widths.get(existing_key, self.bits // 8)
                if write_range is None or existing_range is None:
                    state.memory.pop(existing_key, None)
                    state.memory_widths.pop(existing_key, None)
                    state.memory_types.pop(existing_key, None)
                    continue
                write_root, write_start = write_range
                existing_root, existing_start = existing_range
                if write_root == "va" and existing_root == "va":
                    may_overlap = (
                        write_start < existing_start + existing_width
                        and existing_start < write_start + width
                    )
                elif write_root == existing_root:
                    may_overlap = (
                        write_start < existing_start + existing_width
                        and existing_start < write_start + width
                    )
                else:
                    # Distinct symbolic pointers may alias; keep no fact that a
                    # write through one could have changed through the other.
                    may_overlap = True
                if may_overlap:
                    state.memory.pop(existing_key, None)
                    state.memory_widths.pop(existing_key, None)
                    state.memory_types.pop(existing_key, None)
            state.memory[memory_key] = value
            state.memory_widths[memory_key] = width
            return (f"memory:{memory_key}", value)
        return None

    def _operand_type(self, operand: str, ins: Instruction | None = None) -> str:
        text = operand.strip().lower()
        if text.startswith("byte ptr "):
            return "uint8_t"
        if text.startswith("word ptr "):
            return "uint16_t"
        if text.startswith("dword ptr "):
            return "uint32_t"
        if text.startswith("qword ptr "):
            return "uint64_t"
        if text.startswith(("xmmword ptr ", "ymmword ptr ", "zmmword ptr ")):
            return "vector_t"
        if _parse_int(text) is not None:
            return "int"
        if text.startswith("rip") or ("[" in text and "rip" in text):
            return "uintptr_t"
        reg = _canon_reg(text)
        vector = re.fullmatch(r"([xyz])mm([0-9]|[12][0-9]|3[01])", text)
        if vector:
            return {"x": "vector128_t", "y": "vector256_t", "z": "vector512_t"}[vector.group(1)]
        if reg in {"rsp", "rbp", "rip"}:
            return "uintptr_t"
        if reg:
            return "uint64_t" if self.bits == 64 else "uint32_t"
        return _TYPE_UNKNOWN

    def _set_type(self, dst: str, type_name: str, state: AbstractState) -> None:
        reg = _canon_reg(dst)
        if reg:
            state.reg_types[reg] = type_name
            return
        inner = _mem_inner(dst)
        inner = _canonical_mem_inner(inner)
        if _is_stack_mem(inner) and inner is not None:
            state.stack_types[inner] = type_name
        elif inner is not None:
            state.memory_types[self._memory_key(inner, state)] = type_name

    def _resolved_type(self, operand: str, state: AbstractState, ins: Instruction | None = None) -> str:
        reg = _canon_reg(operand)
        if reg:
            return state.reg_types.get(reg, self._operand_type(operand, ins))
        inner = _mem_inner(operand)
        inner = _canonical_mem_inner(inner)
        if _is_stack_mem(inner) and inner is not None:
            stored_width = state.stack_widths.get(inner)
            if stored_width is not None and stored_width != _operand_width_bytes(operand, self.bits):
                return _TYPE_UNKNOWN
            return state.stack_types.get(inner, self._operand_type(operand, ins))
        if inner is not None:
            memory_key = self._memory_key(inner, state)
            if memory_key in state.memory:
                stored_width = state.memory_widths.get(memory_key)
                if stored_width is not None and stored_width != _operand_width_bytes(operand, self.bits):
                    return _TYPE_UNKNOWN
                return state.memory_types.get(
                    memory_key, "uintptr_t" if "rip" in inner else _TYPE_UNKNOWN
                )
        return self._operand_type(operand, ins)

    def _type_for_write(self, dst: str, source: str | None, state: AbstractState,
                        ins: Instruction) -> str:
        if ins.mnemonic.lower() == "lea":
            return "uintptr_t"
        if ins.mnemonic.lower().startswith("set"):
            return "bool"
        source_type = self._resolved_type(source, state, ins) if source else _TYPE_UNKNOWN
        destination_type = self._operand_type(dst, ins)
        if destination_type == "uintptr_t" or source_type == "uintptr_t":
            return "uintptr_t"
        if destination_type != _TYPE_UNKNOWN and "[" in dst:
            return destination_type
        return source_type if source_type != _TYPE_UNKNOWN else destination_type

    def _call_return_type(self, ins: Instruction, name: str, target: int | None = None) -> str:
        target = ins.direct_target if ins.direct_target is not None else target
        base_name = name.lower().split("!")[-1].split("@", 1)[0]
        signature = get_api_signature(ins.import_name or name)
        if signature and signature["return_type"]:
            return str(signature["return_type"])
        pointer_returns = {
            "malloc", "calloc", "realloc", "aligned_alloc", "_aligned_malloc",
            "strdup", "_strdup", "wcsdup", "heapalloc", "heaprealloc",
            "virtualalloc", "virtualallocex", "localalloc", "globalalloc",
            "cotaskmemalloc", "cotaskmemrealloc", "sysallocstring",
            "memcpy", "memmove", "memset", "strcpy", "strncpy", "strcat",
        }
        void_returns = {
            "free", "_aligned_free", "cotaskmemfree", "sysfreestring",
            "sleep", "abort", "exit", "_exit", "_cexit",
        }
        if base_name in pointer_returns:
            return "uintptr_t"
        if base_name in void_returns:
            return "void"
        if base_name in {"strlen", "virtualquery"}:
            return "size_t"
        if base_name in {"getlasterror", "virtualprotect", "printf", "fprintf", "sprintf", "snprintf", "puts", "fflush", "setvbuf"}:
            return "bool" if base_name == "virtualprotect" else "int32_t" if base_name in {"printf", "fprintf", "sprintf", "snprintf", "puts", "fflush", "setvbuf"} else "uint32_t"
        if target is not None:
            return self.function_return_types.get(
                target, "uint64_t" if self.bits == 64 else "uint32_t"
            )
        return "uint64_t" if self.bits == 64 else "uint32_t"

    def _branch_condition(self, m: str, state: AbstractState) -> str | None:
        known_condition: bool | None = None
        count_alias = {"jcxz": "cx", "jecxz": "ecx", "jrcxz": "rcx"}.get(m)
        if count_alias:
            count = _parse_int(self.resolve(count_alias, state))
            width = {"cx": 16, "ecx": 32, "rcx": 64}[count_alias]
            known_condition = None if count is None else (count & ((1 << width) - 1)) == 0
        elif m in {"loop", "loope", "loopz", "loopne", "loopnz"}:
            alias = "rcx" if self.bits == 64 else "ecx"
            count = _parse_int(self.resolve(alias, state))
            nonzero = None if count is None else count != 0
            if m in {"loope", "loopz"}:
                known_condition = False if nonzero is False else (
                    None if nonzero is None or state.zero is None else state.zero
                )
            elif m in {"loopne", "loopnz"}:
                known_condition = False if nonzero is False else (
                    None if nonzero is None or state.zero is None else not state.zero
                )
            else:
                known_condition = nonzero
        elif m in {"je", "jz"}:
            known_condition = state.zero
        elif m in {"jne", "jnz"}:
            known_condition = None if state.zero is None else not state.zero
        elif m in {"jc", "jb", "jnae"}:
            known_condition = state.carry
        elif m in {"jnc", "jae", "jnb"}:
            known_condition = None if state.carry is None else not state.carry
        elif m in {"jo"}:
            known_condition = state.overflow
        elif m in {"jno"}:
            known_condition = None if state.overflow is None else not state.overflow
        elif m in {"js"}:
            known_condition = state.sign
        elif m in {"jns"}:
            known_condition = None if state.sign is None else not state.sign
        elif m in {"jp", "jpe"}:
            known_condition = state.parity
        elif m in {"jnp", "jpo"}:
            known_condition = None if state.parity is None else not state.parity
        elif m in {"ja", "jnbe"} and state.carry is not None and state.zero is not None:
            known_condition = not state.carry and not state.zero
        elif m in {"jbe", "jna"} and state.carry is not None and state.zero is not None:
            known_condition = state.carry or state.zero
        elif m in {"jg", "jnle"} and state.zero is not None and state.sign is not None and state.overflow is not None:
            known_condition = not state.zero and state.sign == state.overflow
        elif m in {"jge", "jnl"} and state.sign is not None and state.overflow is not None:
            known_condition = state.sign == state.overflow
        elif m in {"jl", "jnge"} and state.sign is not None and state.overflow is not None:
            known_condition = state.sign != state.overflow
        elif m in {"jle", "jng"} and state.zero is not None and state.sign is not None and state.overflow is not None:
            known_condition = state.zero or state.sign != state.overflow
        if known_condition is not None:
            return "1" if known_condition else "0"
        if not state.flags:
            return None
        operation, left, right = state.flags
        if left == _UNKNOWN or right == _UNKNOWN:
            return None
        if operation == "cmp":
            relation = _BRANCH_REL.get(m)
            if relation:
                prefix = "unsigned " if m in _UNSIGNED_BRANCHES else ""
                return f"({prefix}{left} {relation} {right})"
        if operation == "test":
            if m in {"je", "jz"}:
                return f"(({left} & {right}) == 0)"
            if m in {"jne", "jnz"}:
                return f"(({left} & {right}) != 0)"
        if operation in {"add", "sub", "adc", "sbb", "imul", "mul", "and", "or", "xor", "inc", "dec", "neg"}:
            if m in {"je", "jz"}:
                return f"({left} == 0)"
            if m in {"jne", "jnz"}:
                return f"({left} != 0)"
        return None

    def _condition_code(self, suffix: str) -> str | None:
        aliases = {
            "e": "je", "z": "jz", "ne": "jne", "nz": "jnz",
            "b": "jb", "c": "jc", "nae": "jnae", "ae": "jae",
            "nb": "jnb", "nc": "jnc", "a": "ja", "nbe": "jnbe",
            "be": "jbe", "na": "jna", "g": "jg", "nle": "jnle",
            "ge": "jge", "nl": "jnl", "l": "jl", "nge": "jnge",
            "le": "jle", "ng": "jng", "o": "jo", "no": "jno",
            "s": "js", "ns": "jns", "p": "jp", "pe": "jpe",
            "np": "jnp", "po": "jpo",
        }
        return aliases.get(suffix)

    def transfer(self, block: BasicBlock, state_in: AbstractState, function_entry: int,
                 collect: bool = False) -> tuple[AbstractState, list[dict[str, Any]]]:
        state = state_in.clone()
        rows: list[dict[str, Any]] = []
        for ins in block.instructions:
            if not collect:
                self.steps += 1
                if self.steps > self.max_steps_per_function:
                    break
            mnemonic_parts = ins.mnemonic.lower().strip().split()
            prefixes: list[str] = []
            while mnemonic_parts and mnemonic_parts[0] in ({"lock"} | _REPEAT_PREFIXES):
                prefixes.append(mnemonic_parts.pop(0))
            m = mnemonic_parts[-1] if mnemonic_parts else ins.mnemonic.lower()
            raw = split_operands(ins.operands)
            resolved = [self.resolve(x, state, ins) for x in raw]
            resolved_indirect_jump_target: int | None = None
            if (ins.flow == "jump" and ins.direct_target is None and raw
                    and "*" not in raw[0] and ins.jump_table_case_count is None):
                target_text = resolved[0].strip()
                static_target = re.fullmatch(r"static_ptr\((0x[0-9a-f]+|\d+)\)", target_text, re.I)
                candidate = _parse_int(static_target.group(1)) if static_target else _parse_int(target_text)
                if candidate in self.known_code_addresses:
                    resolved_indirect_jump_target = candidate
            assigned = None
            additional_writes: list[dict[str, str]] = []
            condition = None
            write_type = None
            resolved_call_target: int | None = None
            resolved_call_import: str | None = None
            resolved_call_provenance: str | None = None
            call_argument_values: list[str] | None = None
            call_stack_memory_values: list[str] | None = None
            call_argument_types: list[str] | None = None
            if m in {"finit", "fninit"}:
                state.x87_stack = []
            elif m in {"fabs", "fchs"}:
                old_value = state.x87_stack[0] if state.x87_stack else _UNKNOWN
                if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", old_value):
                    try:
                        number = Decimal(old_value)
                        value = str(abs(number) if m == "fabs" else -number)
                        if "." not in value and "E" not in value and "e" not in value:
                            value += ".0"
                    except InvalidOperation:
                        value = _UNKNOWN
                elif old_value == _UNKNOWN:
                    value = _UNKNOWN
                else:
                    value = f"{'abs' if m == 'fabs' else 'negate'}_x87({old_value})"
                if state.x87_stack is not None:
                    if state.x87_stack:
                        state.x87_stack[0] = value
                    else:
                        state.x87_stack = None
                assigned = ("ST0", value)
                write_type = "long double"
            elif m == "fild" and raw:
                number = _parse_int(resolved[0])
                source_width = _operand_width_bytes(raw[0], self.bits) * 8
                if number is not None:
                    sign_bit = 1 << (source_width - 1)
                    if number & sign_bit:
                        number -= 1 << source_width
                    value = f"{number}.0"
                else:
                    value = f"signed_int_to_x87({resolved[0]})"
                if state.x87_stack is not None:
                    if len(state.x87_stack) < 8:
                        state.x87_stack.insert(0, value)
                    else:
                        state.x87_stack = None
                assigned = ("ST0", value)
                write_type = "long double"
            elif m in {"fist", "fistp", "fisttp"} and raw:
                value = state.x87_stack[0] if state.x87_stack else _UNKNOWN
                conversion = "truncate_to_int" if m == "fisttp" else "round_to_int_using_x87_control_word"
                converted = _UNKNOWN if value == _UNKNOWN else f"{conversion}({value})"
                assigned = self._assign(raw[0], converted, state)
                write_type = {16: "int16_t", 32: "int32_t", 64: "int64_t"}.get(
                    _operand_width_bytes(raw[0], self.bits) * 8, "int"
                )
                if m in {"fistp", "fisttp"}:
                    if state.x87_stack:
                        state.x87_stack.pop(0)
                    else:
                        state.x87_stack = None
                    additional_writes.append({"target": "ST0", "value": state.x87_stack[0] if state.x87_stack else _UNKNOWN})
            elif m in {"fld1", "fldz", "fldpi"}:
                value = {"fld1": "1.0", "fldz": "0.0", "fldpi": "x87_pi"}[m]
                if state.x87_stack is not None:
                    if len(state.x87_stack) < 8:
                        state.x87_stack.insert(0, value)
                    else:
                        state.x87_stack = None
                assigned = ("ST0", value)
                write_type = "long double"
            elif m == "fld" and raw:
                index = _x87_index(raw[0])
                value = (state.x87_stack[index] if index is not None and state.x87_stack is not None
                         and index < len(state.x87_stack) else resolved[0])
                if state.x87_stack is not None:
                    if len(state.x87_stack) < 8:
                        state.x87_stack.insert(0, value)
                    else:
                        state.x87_stack = None
                assigned = ("ST0", value)
                write_type = "long double"
            elif m in {"fst", "fstp"} and raw:
                value = state.x87_stack[0] if state.x87_stack else _UNKNOWN
                index = _x87_index(raw[0])
                if index is not None:
                    if state.x87_stack is not None and index < len(state.x87_stack):
                        state.x87_stack[index] = value
                    assigned = (f"ST{index}", value)
                else:
                    assigned = self._assign(raw[0], value, state)
                    write_type = "long double"
                if m == "fstp":
                    if state.x87_stack:
                        state.x87_stack.pop(0)
                    else:
                        state.x87_stack = None
                    additional_writes.append({"target": "ST0", "value": state.x87_stack[0] if state.x87_stack else _UNKNOWN})
            elif m == "fxch":
                index = _x87_index(raw[0]) if raw else 1
                index = 1 if index is None else index
                if state.x87_stack is not None and index < len(state.x87_stack):
                    state.x87_stack[0], state.x87_stack[index] = state.x87_stack[index], state.x87_stack[0]
                else:
                    state.x87_stack = None
                assigned = ("ST0", state.x87_stack[0] if state.x87_stack else _UNKNOWN)
                additional_writes.append({f"target": f"ST{index}", "value": state.x87_stack[index] if state.x87_stack and index < len(state.x87_stack) else _UNKNOWN})
            elif re.fullmatch(r"fcmov(?:b|e|be|u|nb|ne|nbe|nu)", m) and len(raw) >= 2:
                index = _x87_index(raw[1])
                index = 1 if index is None else index
                conditions = {"b": "jb", "e": "je", "be": "jbe", "u": "jp",
                              "nb": "jnb", "ne": "jne", "nbe": "jnbe", "nu": "jnp"}
                known = self._branch_condition(conditions[m[5:]], state)
                old_value = state.x87_stack[0] if state.x87_stack else _UNKNOWN
                source_value = state.x87_stack[index] if state.x87_stack and index < len(state.x87_stack) else _UNKNOWN
                value = source_value if known == "1" else old_value if known == "0" else (
                    old_value if old_value == source_value else f"select(condition:{m[5:]}, {source_value}, {old_value})"
                )
                if state.x87_stack is not None and state.x87_stack:
                    state.x87_stack[0] = value
                else:
                    state.x87_stack = None
                assigned = ("ST0", value)
                write_type = "long double"
            elif m in {"fadd", "faddp", "fsub", "fsubp", "fsubr", "fsubrp", "fmul", "fmulp", "fdiv", "fdivp", "fdivr", "fdivrp"}:
                pop_after = m.endswith("p")
                base = m[:-1] if pop_after else m
                operation = {"fadd": "+", "fmul": "*", "fsub": "-", "fsubr": "-", "fdiv": "/", "fdivr": "/"}[base]
                if len(raw) >= 2:
                    destination_index = _x87_index(raw[0])
                    source_index = _x87_index(raw[1])
                else:
                    destination_index = 1 if pop_after else 0
                    source_index = 0 if pop_after else (_x87_index(raw[0]) if raw else None)
                destination_value = (
                    state.x87_stack[destination_index]
                    if state.x87_stack is not None and destination_index is not None
                    and destination_index < len(state.x87_stack) else _UNKNOWN
                )
                if source_index is not None:
                    source_value = (
                        state.x87_stack[source_index]
                        if state.x87_stack is not None and source_index < len(state.x87_stack) else _UNKNOWN
                    )
                elif raw:
                    source_value = resolved[-1]
                else:
                    source_value = _UNKNOWN
                lhs, rhs = (source_value, destination_value) if base in {"fsubr", "fdivr"} else (destination_value, source_value)
                folded = (
                    _fold_exact_x87_integer_arithmetic(lhs, rhs, operation)
                    if _UNKNOWN not in {lhs, rhs} else None
                )
                result = (
                    _UNKNOWN if _UNKNOWN in {lhs, rhs}
                    else folded if folded is not None
                    else f"({lhs} {operation} {rhs})"
                )
                if len(result) > 160:
                    result = _UNKNOWN
                if state.x87_stack is not None and destination_index is not None and destination_index < len(state.x87_stack):
                    state.x87_stack[destination_index] = result
                else:
                    state.x87_stack = None
                assigned = (f"ST{destination_index or 0}", result)
                if pop_after:
                    if state.x87_stack:
                        state.x87_stack.pop(0)
                    else:
                        state.x87_stack = None
                    additional_writes.append({"target": "ST0", "value": state.x87_stack[0] if state.x87_stack else _UNKNOWN})
                write_type = "long double"
            elif m in {"fcom", "fucom", "fcomp", "fucomp", "fcompp", "fucompp", "fcomi", "fucomi", "fcomip", "fucomip"}:
                pop_count = 2 if m.endswith("pp") else 1 if m.endswith("p") else 0
                if m in {"fcomi", "fucomi", "fcomip", "fucomip"}:
                    compare_index = _x87_index(raw[-1]) if raw else 1
                    left = _parse_x87_number(state.x87_stack[0]) if state.x87_stack else None
                    right = (_parse_x87_number(state.x87_stack[compare_index])
                             if state.x87_stack and compare_index is not None
                             and compare_index < len(state.x87_stack) else None)
                    if left is not None and right is not None:
                        state.carry = left < right
                        state.zero = left == right
                        state.parity = False
                        state.sign = state.overflow = state.auxiliary = False
                        state.flags = ("cmp", str(left), str(right))
                    else:
                        state.flags = None
                        state.carry = state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
                if state.x87_stack is not None and len(state.x87_stack) >= pop_count:
                    del state.x87_stack[:pop_count]
                elif pop_count:
                    state.x87_stack = None
                assigned = ("X87_STATUS", _UNKNOWN)
            elif re.fullmatch(r"set(?:e|z|ne|nz|b|c|nae|ae|nb|nc|a|nbe|be|na|g|nle|ge|nl|l|nge|le|ng|o|no|s|ns|p|pe|np|po)", m) and raw:
                suffix = m[3:]
                branch_mnemonic = self._condition_code(suffix)
                known = self._branch_condition(branch_mnemonic, state) if branch_mnemonic else None
                value = known if known is not None else f"bool(condition:{suffix})"
                assigned = self._assign(raw[0], value, state)
                write_type = "bool" if _canon_reg(raw[0]) else None
            elif re.fullmatch(r"cmov(?:e|z|ne|nz|b|c|nae|ae|nb|nc|a|nbe|be|na|g|nle|ge|nl|l|nge|le|ng|o|no|s|ns|p|pe|np|po)", m) and len(raw) >= 2:
                suffix = m[4:]
                branch_mnemonic = self._condition_code(suffix)
                known = self._branch_condition(branch_mnemonic, state) if branch_mnemonic else None
                if known == "1":
                    value = resolved[1]
                elif known == "0":
                    value = resolved[0]
                elif resolved[0] == resolved[1]:
                    value = resolved[0]
                else:
                    value = f"select(condition:{suffix}, {resolved[1]}, {resolved[0]})"
                assigned = self._assign(raw[0], value, state)
                write_type = self._operand_type(raw[0], ins)
            elif m in _STRING_KINDS:
                kind, _width = _STRING_KINDS[m]
                repeat_prefix = next((prefix for prefix in prefixes if prefix in _REPEAT_PREFIXES), None)
                conditional_repeat = (
                    repeat_prefix in {"repe", "repz", "repne", "repnz"}
                    or (repeat_prefix == "rep" and kind in {"compare", "scan"})
                )
                count_value = _parse_int(state.regs.get("rcx", _UNKNOWN))
                string_iterations_executed: int | None = None
                string_compare_known = False
                if kind in {"compare", "scan"} and state.direction is not None:
                    requested = count_value if repeat_prefix else 1
                    source_pointer = _parse_int(state.regs.get("rsi", _UNKNOWN))
                    destination_pointer = _parse_int(state.regs.get("rdi", _UNKNOWN))
                    accumulator = _parse_int(self.resolve(
                        {8: "al", 16: "ax", 32: "eax", 64: "rax"}[_width], state, ins
                    )) if kind == "scan" else None
                    if (requested is not None and 0 <= requested <= 256
                            and destination_pointer is not None
                            and (kind == "scan" or source_pointer is not None)):
                        size = _width // 8
                        size_name = {1: "byte", 2: "word", 4: "dword", 8: "qword"}[size]
                        step = -size if state.direction else size
                        address_mask = (1 << self.bits) - 1
                        executed = 0
                        known = True
                        for iteration in range(requested):
                            destination_address = (destination_pointer + iteration * step) & address_mask
                            destination_operand = f"{size_name} ptr [0x{destination_address:X}]"
                            right = _parse_int(self.resolve(destination_operand, state, ins))
                            if kind == "compare":
                                source_address = (source_pointer + iteration * step) & address_mask
                                source_operand = f"{size_name} ptr [0x{source_address:X}]"
                                left = _parse_int(self.resolve(source_operand, state, ins))
                            else:
                                left = accumulator
                            if left is None or right is None:
                                known = False
                                break
                            (result, state.carry, state.parity, state.auxiliary, state.zero,
                             state.sign, state.overflow) = _arithmetic_condition_flags(
                                "cmp", left, right, _width
                            )
                            executed += 1
                            state.flags = ("cmp", _format_int(left), _format_int(right))
                            if not conditional_repeat:
                                continue
                            should_repeat = (
                                bool(state.zero) if repeat_prefix in {"rep", "repe", "repz"}
                                else not bool(state.zero)
                            )
                            if not should_repeat:
                                break
                        if known:
                            string_iterations_executed = executed
                            string_compare_known = executed > 0
                if not (repeat_prefix and count_value == 0):
                    writes_memory = kind in {"move", "store", "input"}
                    if writes_memory:
                        iterations_for_write = count_value if repeat_prefix else 1
                        source_pointer = _parse_int(state.regs.get("rsi", _UNKNOWN))
                        destination_pointer = _parse_int(state.regs.get("rdi", _UNKNOWN))
                        precise_string_write = (
                            iterations_for_write is not None
                            and 0 < iterations_for_write <= 256
                            and state.direction is not None and destination_pointer is not None
                            and (kind != "move" or source_pointer is not None)
                        )
                        if precise_string_write:
                            size = _width // 8
                            size_name = {1: "byte", 2: "word", 4: "dword", 8: "qword"}[size]
                            step = -size if state.direction else size
                            address_mask = (1 << self.bits) - 1
                            accumulator = {8: "al", 16: "ax", 32: "eax", 64: "rax"}.get(_width)
                            for iteration in range(iterations_for_write):
                                destination_address = (destination_pointer + iteration * step) & address_mask
                                destination_operand = f"{size_name} ptr [0x{destination_address:X}]"
                                if kind == "move":
                                    source_address = (source_pointer + iteration * step) & address_mask
                                    source_operand = f"{size_name} ptr [0x{source_address:X}]"
                                    value = self.resolve(source_operand, state, ins)
                                    self._assign(destination_operand, value, state, source_operand)
                                else:
                                    value = self.resolve(accumulator, state, ins) if kind == "store" else _UNKNOWN
                                    self._assign(destination_operand, value, state)
                        else:
                            state.memory.clear()
                            state.memory_widths.clear()
                            state.memory_types.clear()
                            state.stack.clear()
                            state.stack_widths.clear()
                            state.stack_types.clear()
                    affected_registers = {
                        "move": ["rsi", "rdi"], "store": ["rdi"], "load": ["rax", "rsi"],
                        "compare": ["rsi", "rdi"], "scan": ["rdi"],
                        "input": ["rdi"], "output": ["rsi"],
                    }[kind]
                    iterations = (
                        string_iterations_executed
                        if string_iterations_executed is not None
                        else count_value if repeat_prefix and not conditional_repeat
                        else 1 if not repeat_prefix else None
                    )
                    delta = None
                    if iterations is not None and state.direction is not None:
                        delta = iterations * (_width // 8) * (-1 if state.direction else 1)
                    pointer_registers = {
                        "move": ["rsi", "rdi"], "store": ["rdi"], "load": ["rsi"],
                        "compare": ["rsi", "rdi"], "scan": ["rdi"],
                        "input": ["rdi"], "output": ["rsi"],
                    }[kind]
                    new_values: dict[str, str] = {}
                    for register in affected_registers:
                        if register == "rax" or register == "rcx":
                            new_values[register] = _UNKNOWN
                        elif register in pointer_registers and delta is not None:
                            pointer = _parse_int(state.regs.get(register, _UNKNOWN))
                            if pointer is None:
                                new_values[register] = _UNKNOWN
                            else:
                                pointer_mask = (1 << self.bits) - 1
                                new_values[register] = _format_int((pointer + delta) & pointer_mask)
                        else:
                            new_values[register] = _UNKNOWN
                    for register, value in new_values.items():
                        state.regs[register] = value
                        _clear_register_slices(state, register)
                        if value == _UNKNOWN:
                            state.reg_types.pop(register, None)
                    if repeat_prefix:
                        # Unconditional REP runs until the count reaches zero;
                        # REPE/REPNE can stop early on the comparison condition.
                        if conditional_repeat:
                            state.regs["rcx"] = (
                                _format_int(max(0, count_value - string_iterations_executed))
                                if string_iterations_executed is not None and count_value is not None
                                else _UNKNOWN
                            )
                        else:
                            state.regs["rcx"] = "0"
                        _clear_register_slices(state, "rcx")
                        state.reg_types.pop("rcx", None)
                    if kind in {"compare", "scan"}:
                        if not string_compare_known:
                            state.flags = None
                            state.carry = state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
                    if affected_registers:
                        primary = affected_registers[0]
                        assigned = (primary, state.regs[primary])
                        additional_writes.extend(
                            {"target": register, "value": state.regs[register]}
                            for register in affected_registers[1:]
                        )
                    if repeat_prefix:
                        additional_writes.append({"target": "rcx", "value": state.regs["rcx"]})
                    if kind in {"compare", "scan"}:
                        additional_writes.append({
                            "target": "flags",
                            "value": {
                                "CF": state.carry, "PF": state.parity, "AF": state.auxiliary,
                                "ZF": state.zero, "SF": state.sign, "OF": state.overflow,
                            },
                        })
            elif m in {"cld", "std"}:
                state.direction = m == "std"
            elif re.fullmatch(r"v?pextr[bwdq]|v?extractps", m) and len(raw) >= 3:
                lane_bits = {"b": 8, "w": 16, "d": 32, "q": 64}.get(m[-1], 32)
                index = _parse_int(resolved[2])
                vector = _parse_int(resolved[1])
                vector_bits = _operand_width_bytes(raw[1], self.bits) * 8
                lane_count = vector_bits // lane_bits
                if index is None or vector is None or lane_count <= 0:
                    value = _UNKNOWN
                else:
                    index &= lane_count - 1
                    value = _format_int((vector >> (index * lane_bits)) & ((1 << lane_bits) - 1))
                assigned = self._assign(raw[0], value, state)
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?pinsr[bwdq]", m) and len(raw) >= 3:
                lane_bits = {"b": 8, "w": 16, "d": 32, "q": 64}[m[-1]]
                destination_bits = _operand_width_bytes(raw[0], self.bits) * 8
                lane_count = destination_bits // lane_bits
                index = _parse_int(resolved[-1])
                vector = _parse_int(resolved[1] if m.startswith("v") else resolved[0])
                scalar = _parse_int(resolved[2] if m.startswith("v") else resolved[1])
                if index is None or vector is None or scalar is None or lane_count <= 0:
                    value = _UNKNOWN
                else:
                    index &= lane_count - 1
                    shift = index * lane_bits
                    lane_mask = ((1 << lane_bits) - 1) << shift
                    value = _format_int((vector & ~lane_mask) | ((scalar << shift) & lane_mask))
                assigned = self._assign(raw[0], value, state, zero_upper_vector=m.startswith("v"))
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?ptest", m) and len(raw) >= 2:
                left, right = _parse_int(resolved[0]), _parse_int(resolved[1])
                width = _operand_width_bytes(raw[0], self.bits) * 8
                mask = (1 << width) - 1 if width > 0 else 0
                if left is None or right is None or width <= 0:
                    state.carry = state.zero = None
                else:
                    state.zero = (left & right & mask) == 0
                    state.carry = ((~left & right) & mask) == 0
                state.overflow = state.sign = state.parity = state.auxiliary = None
                state.flags = None
            elif re.fullmatch(r"v?palignr|v?pshufb|v?pshuf[hl]w|v?pshufd|v?shufps|v?shufpd", m):
                is_vex = m.startswith("v")
                vector_bits = _operand_width_bytes(raw[0], self.bits) * 8 if raw else 0
                if m.endswith("palignr"):
                    first = resolved[1] if is_vex else resolved[0]
                    second = resolved[2] if is_vex and len(resolved) >= 3 else resolved[1] if len(resolved) >= 2 else _UNKNOWN
                    control = resolved[3] if is_vex and len(resolved) >= 4 else resolved[2] if len(resolved) >= 3 else _UNKNOWN
                elif m.endswith("pshufb"):
                    first = resolved[1] if is_vex else resolved[0]
                    second = resolved[2] if is_vex and len(resolved) >= 3 else resolved[1] if len(resolved) >= 2 else _UNKNOWN
                    control = "0"
                elif m.endswith(("pshufd", "pshuflw", "pshufhw")):
                    first = resolved[1] if len(resolved) >= 2 else _UNKNOWN
                    second = "0"
                    control = resolved[2] if len(resolved) >= 3 else _UNKNOWN
                else:
                    first = resolved[1] if is_vex and len(resolved) >= 2 else resolved[0] if resolved else _UNKNOWN
                    second = resolved[2] if is_vex and len(resolved) >= 3 else resolved[1] if len(resolved) >= 2 else _UNKNOWN
                    control_index = 3 if is_vex else 2
                    control = resolved[control_index] if len(resolved) > control_index else _UNKNOWN
                value = _fold_vector_shuffle(m, first, second, control, vector_bits)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=is_vex)
                write_type = self._operand_type(raw[0], ins)
            elif m == "andn" and len(raw) >= 3:
                width = _operand_width_bytes(raw[0], self.bits) * 8
                left, right = _parse_int(resolved[1]), _parse_int(resolved[2])
                mask = (1 << width) - 1 if width > 0 else 0
                result = None if left is None or right is None or width <= 0 else (~left & right) & mask
                value = _UNKNOWN if result is None else _format_int(result)
                assigned = self._assign(raw[0], value, state)
                write_type = self._operand_type(raw[0], ins)
                state.carry = state.overflow = False
                state.zero = None if result is None else result == 0
                state.sign = None if result is None else bool(result & (1 << (width - 1)))
                state.parity = state.auxiliary = None
                state.flags = None
            elif m in {"blsi", "blsmsk", "blsr"} and len(raw) >= 2:
                width = _operand_width_bytes(raw[0], self.bits) * 8
                source = _parse_int(resolved[1])
                mask = (1 << width) - 1 if width > 0 else 0
                if source is None or width <= 0:
                    result = None
                else:
                    source &= mask
                    result = {
                        "blsi": source & -source,
                        "blsmsk": source ^ ((source - 1) & mask),
                        "blsr": source & ((source - 1) & mask),
                    }[m] & mask
                value = _UNKNOWN if result is None else _format_int(result)
                assigned = self._assign(raw[0], value, state)
                write_type = self._operand_type(raw[0], ins)
                state.carry = None if source is None else source == 0
                state.overflow = False
                state.zero = None if result is None else result == 0
                state.sign = None if result is None or width <= 0 else bool(result & (1 << (width - 1)))
                state.parity = state.auxiliary = None
                state.flags = None
            elif m in {"popcnt", "lzcnt", "tzcnt", "bsf", "bsr"} and len(raw) >= 2:
                width = _operand_width_bytes(raw[1], self.bits) * 8
                source = _parse_int(resolved[1])
                mask = (1 << width) - 1 if width > 0 else 0
                if source is None or width <= 0:
                    value = _UNKNOWN
                else:
                    source &= mask
                    if m == "popcnt":
                        result = source.bit_count()
                    elif m in {"lzcnt", "bsr"}:
                        result = width if source == 0 else width - source.bit_length()
                        if m == "bsr" and source:
                            result = source.bit_length() - 1
                    elif m in {"tzcnt", "bsf"}:
                        result = width if source == 0 else (source & -source).bit_length() - 1
                    value = _UNKNOWN if source == 0 and m in {"bsf", "bsr"} else _format_int(result)
                assigned = self._assign(raw[0], value, state)
                write_type = self._operand_type(raw[0], ins)
                if source is None or width <= 0:
                    state.carry = state.zero = None
                    state.sign = state.overflow = state.parity = state.auxiliary = None
                elif m == "popcnt":
                    state.carry = state.sign = state.overflow = state.parity = state.auxiliary = False
                    state.zero = source == 0
                elif m in {"lzcnt", "tzcnt"}:
                    state.carry = source == 0
                    state.zero = result == 0
                    state.sign = state.overflow = state.parity = state.auxiliary = None
                else:
                    state.zero = source == 0
                    state.carry = state.sign = state.overflow = state.parity = state.auxiliary = None
                state.flags = None
            elif re.fullmatch(r"v?(?:movmskps|movmskpd|pmovmskb)", m) and len(raw) >= 2:
                source_width = _operand_width_bytes(raw[1], self.bits) * 8
                value = _fold_vector_movemask(m, resolved[1], source_width)
                assigned = self._assign(raw[0], value, state)
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?(?:blendps|blendpd|pblendw|pblendd)", m) and len(raw) >= 3:
                if m.startswith("v") and len(raw) >= 4:
                    left, right, control = resolved[1], resolved[2], resolved[3]
                else:
                    left, right, control = resolved[0], resolved[1], resolved[2]
                vector_bits = _operand_width_bytes(raw[0], self.bits) * 8
                value = _fold_vector_shuffle(m, left, right, control, vector_bits)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=m.startswith("v"))
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?pblendvb", m) and len(raw) >= 2:
                if m.startswith("v") and len(raw) >= 4:
                    left, right, mask = resolved[1], resolved[2], resolved[3]
                else:
                    left, right = resolved[0], resolved[1]
                    mask = self.resolve("xmm0", state, ins)
                vector_bits = _operand_width_bytes(raw[0], self.bits) * 8
                value = _fold_vector_blend_variable(left, right, mask, vector_bits)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=m.startswith("v"))
                write_type = self._operand_type(raw[0], ins)
            elif m in {
                "movd", "movq", "vmovd", "vmovq", "movdqa", "movdqu",
                "movaps", "movups", "movapd", "movupd", "vmovdqa", "vmovdqu",
                "vmovaps", "vmovups", "vmovapd", "vmovupd",
            } and len(raw) >= 2:
                value = resolved[1]
                destination_bits = _operand_width_bytes(raw[0], self.bits) * 8
                source_bits = _operand_width_bytes(raw[1], self.bits) * 8
                if m in {"movd", "vmovd", "movq", "vmovq"}:
                    number = _parse_int(value)
                    if number is not None:
                        value = _format_int(number & ((1 << min(source_bits, destination_bits)) - 1))
                    else:
                        value = _UNKNOWN
                assigned = self._assign(raw[0], value, state, raw[1], zero_upper_vector=m.startswith("v"))
                write_type = self._operand_type(raw[0], ins)
            elif (m in {"pxor", "vpxor", "por", "vpor", "pand", "vpand", "pandn", "vpandn"}
                  or re.fullmatch(r"v?pmul(?:lw|hw|huw|ld|dq|udq)", m)
                  or re.fullmatch(r"v?pack(?:sswb|ssdw|uswb|usdw)", m)
                  or re.fullmatch(r"v?(?:pmaddwd|pmaddubsw|psadbw)", m)
                  or re.fullmatch(r"v?ph(?:add|sub)(?:sw|w|d)", m)
                  or re.fullmatch(r"v?psign[bwd]", m)
                  or re.fullmatch(r"v?pabs[bwd]", m)
                  or re.fullmatch(r"v?punpck[lh](?:bw|wd|dq|qdq)", m)
                  or re.fullmatch(r"v?p(?:add|sub)[bwdq]", m)
                  or re.fullmatch(r"v?ps(?:ll|rl|ra)[wdq]", m)
                  or re.fullmatch(r"v?ps(?:ll|rl)dq", m)
                  or re.fullmatch(r"v?p(?:add|sub)(?:us|s)[bw]", m)
                  or re.fullmatch(r"v?pavg[bw]", m)
                  or re.fullmatch(r"v?pcmp(?:eq|gt)[bwdq]", m)
                  or re.fullmatch(r"v?p(?:min|max)[us][bwd]", m)) and len(raw) >= 2:
                if m.startswith("v") and len(raw) >= 3:
                    left, right = resolved[1], resolved[2]
                else:
                    left, right = resolved[0], resolved[1]
                vector_bits = _operand_width_bytes(raw[0], self.bits) * 8
                value = _fold_packed_integer(m, left, right, vector_bits)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=m.startswith("v"))
                write_type = self._operand_type(raw[0], ins)
            elif m == "mov" and len(raw) == 2:
                assigned = self._assign(raw[0], resolved[1], state, raw[1])
                write_type = self._type_for_write(raw[0], raw[1], state, ins)
            elif m in {"movzx", "movsx", "movsxd"} and len(raw) == 2:
                source_value = _parse_int(resolved[1])
                source_bits = _operand_width_bytes(raw[1], self.bits) * 8
                destination_bits = _operand_width_bytes(raw[0], self.bits) * 8
                if source_value is None or source_bits <= 0 or destination_bits <= 0:
                    extended = _UNKNOWN
                else:
                    source_mask = (1 << source_bits) - 1
                    extended = source_value & source_mask
                    if m in {"movsx", "movsxd"} and extended & (1 << (source_bits - 1)):
                        extended -= 1 << source_bits
                    extended &= (1 << destination_bits) - 1
                    extended = _format_int(extended)
                assigned = self._assign(raw[0], extended, state, raw[1])
                write_type = self._operand_type(raw[0], ins)
            elif m == "not" and raw:
                width_bits = _operand_width_bytes(raw[0], self.bits) * 8
                value = _parse_int(resolved[0])
                result = _UNKNOWN if value is None else _format_int((~value) & ((1 << width_bits) - 1))
                assigned = self._assign(raw[0], result, state)
                write_type = self._operand_type(raw[0], ins)
                # NOT changes no arithmetic flags.
            elif m == "bswap" and raw and _canon_reg(raw[0]):
                width_bits = _operand_width_bytes(raw[0], self.bits) * 8
                value = _parse_int(resolved[0])
                if width_bits not in {32, 64} or value is None:
                    result = _UNKNOWN
                else:
                    byte_count = width_bits // 8
                    mask = (1 << width_bits) - 1
                    result_value = int.from_bytes((value & mask).to_bytes(byte_count, "little"), "big")
                    result = _format_int(result_value)
                assigned = self._assign(raw[0], result, state)
                write_type = self._operand_type(raw[0], ins)
                # BSWAP changes no arithmetic flags.
            elif m in {"rol", "ror", "rcl", "rcr"} and len(raw) >= 2:
                width_bits = _operand_width_bytes(raw[0], self.bits) * 8
                value_number = _parse_int(resolved[0])
                count_number = _parse_int(resolved[1])
                count_mask = 0x3F if width_bits == 64 else 0x1F
                masked_count = None if count_number is None else count_number & count_mask
                if masked_count is None:
                    effective_count = None
                elif m in {"rol", "ror"}:
                    effective_count = masked_count % width_bits
                else:
                    effective_count = masked_count % (width_bits + 1)
                if effective_count == 0:
                    # A zero effective rotate count preserves every flag.
                    result = resolved[0]
                else:
                    result = _UNKNOWN
                    result_value: int | None = None
                    mask = (1 << width_bits) - 1
                    if value_number is None or effective_count is None:
                        result = _UNKNOWN
                        state.carry = None
                        state.overflow = None
                    else:
                        value = value_number & mask
                        count = effective_count
                        if m == "rol":
                            result_value = ((value << count) | (value >> (width_bits - count))) & mask
                            state.carry = bool(result_value & 1)
                            msb = (result_value >> (width_bits - 1)) & 1
                            state.overflow = bool(msb ^ int(state.carry)) if count == 1 else None
                        elif m == "ror":
                            result_value = ((value >> count) | (value << (width_bits - count))) & mask
                            state.carry = bool((result_value >> (width_bits - 1)) & 1)
                            top_two = (result_value >> (width_bits - 2)) & 0b11
                            state.overflow = bool(((top_two >> 1) ^ (top_two & 1))) if count == 1 else None
                        elif state.carry is None:
                            state.overflow = None
                        else:
                            combined_width = width_bits + 1
                            combined_mask = (1 << combined_width) - 1
                            combined = ((int(state.carry) << width_bits) | value) & combined_mask
                            if m == "rcl":
                                rotated = ((combined << count) | (combined >> (combined_width - count))) & combined_mask
                                result_value = rotated & mask
                                state.carry = bool((rotated >> width_bits) & 1)
                                msb = (result_value >> (width_bits - 1)) & 1
                                state.overflow = bool(msb ^ int(state.carry)) if count == 1 else None
                            else:
                                rotated = ((combined >> count) | (combined << (combined_width - count))) & combined_mask
                                result_value = rotated & mask
                                state.carry = bool((rotated >> width_bits) & 1)
                                top_two = (result_value >> (width_bits - 2)) & 0b11
                                state.overflow = bool((top_two >> 1) ^ (top_two & 1)) if count == 1 else None
                        if result_value is not None:
                            result = _format_int(result_value)
                    state.flags = None
                assigned = self._assign(raw[0], result, state)
                write_type = self._operand_type(raw[0], ins)
            elif m in {"shld", "shrd"} and len(raw) == 3:
                width_bits = _operand_width_bytes(raw[0], self.bits) * 8
                mask = (1 << width_bits) - 1
                destination = _parse_int(resolved[0])
                source = _parse_int(resolved[1])
                raw_count = _parse_int(resolved[2])
                count_mask = 0x3F if width_bits == 64 else 0x1F
                count = None if raw_count is None else raw_count & count_mask
                if count == 0:
                    result = resolved[0]
                elif destination is None or source is None or count is None or count > width_bits:
                    result = _UNKNOWN
                    state.carry = state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
                    state.flags = None
                else:
                    old_value = destination & mask
                    source_value = source & mask
                    if m == "shld":
                        shifted = ((old_value << count) | (source_value >> (width_bits - count))) & mask
                        state.carry = bool((old_value >> (width_bits - count)) & 1)
                        state.overflow = bool(((shifted >> (width_bits - 1)) & 1) ^ int(state.carry)) if count == 1 else None
                    else:
                        shifted = ((old_value >> count) | (source_value << (width_bits - count))) & mask
                        state.carry = bool((old_value >> (count - 1)) & 1)
                        state.overflow = bool(((old_value >> (width_bits - 1) ^ shifted >> (width_bits - 1)) & 1)) if count == 1 else None
                    result = _format_int(shifted)
                    state.zero, state.sign, state.parity = _result_condition_flags(shifted, width_bits)
                    state.auxiliary = None
                    state.flags = (m, result, str(count))
                assigned = self._assign(raw[0], result, state)
                write_type = self._operand_type(raw[0], ins)
            elif m == "xchg" and len(raw) == 2:
                if all(_canon_reg(operand) for operand in raw):
                    left, right = resolved[0], resolved[1]
                    self._assign(raw[0], right, state)
                    self._assign(raw[1], left, state)
                    assigned = (_canon_reg(raw[0]) or raw[0].lower(), right)
                    additional_writes.append({"target": _canon_reg(raw[1]) or raw[1].lower(), "value": left})
                else:
                    # XCHG with memory is implicitly locked. Until a memory-SSA
                    # transfer is available, invalidate every possibly aliased fact.
                    state.memory.clear()
                    state.memory_widths.clear()
                    state.memory_types.clear()
                    state.stack.clear()
                    state.stack_widths.clear()
                    state.stack_types.clear()
                    for operand in raw:
                        reg = _canon_reg(operand)
                        if reg:
                            state.regs[reg] = _UNKNOWN
                            _clear_register_slices(state, reg)
                            state.reg_types.pop(reg, None)
            elif m in {"xadd", "cmpxchg"} and len(raw) >= 2:
                register_form = all(_canon_reg(operand) for operand in raw[:2])
                memory_form = _mem_inner(raw[0]) is not None and _canon_reg(raw[1]) is not None
                width_bits = _operand_width_bytes(raw[0], self.bits) * 8
                mask = (1 << width_bits) - 1
                left_number = _parse_int(resolved[0])
                right_number = _parse_int(resolved[1])
                if (register_form or memory_form) and m == "xadd" and left_number is not None and right_number is not None:
                    left_value, right_value = left_number & mask, right_number & mask
                    result = _format_int((left_value + right_value) & mask)
                    assigned = self._assign(raw[0], result, state)
                    if register_form and _canon_reg(raw[0]) != _canon_reg(raw[1]):
                        additional_writes.append({
                            "target": _canon_reg(raw[1]) or raw[1].lower(),
                            "value": _format_int(left_value),
                        })
                        self._assign(raw[1], _format_int(left_value), state)
                    elif memory_form:
                        source_register = _canon_reg(raw[1])
                        self._assign(raw[1], _format_int(left_value), state)
                        additional_writes.append({
                            "target": source_register or raw[1].lower(),
                            "value": _format_int(left_value),
                        })
                    write_type = self._operand_type(raw[0], ins)
                    (_result, state.carry, state.parity, state.auxiliary, state.zero,
                     state.sign, state.overflow) = _arithmetic_condition_flags(
                        "add", left_value, right_value, width_bits, 0
                    )
                    state.flags = ("add", result, "0")
                elif (register_form or memory_form) and m == "cmpxchg":
                    accumulator = {8: "al", 16: "ax", 32: "eax", 64: "rax"}.get(width_bits)
                    accumulator_number = _parse_int(self.resolve(accumulator or "rax", state, ins))
                    if accumulator_number is not None and left_number is not None and right_number is not None:
                        accumulator_value = accumulator_number & mask
                        destination_value = left_number & mask
                        source_value = right_number & mask
                        equal = accumulator_value == destination_value
                        (_result, state.carry, state.parity, state.auxiliary, state.zero,
                         state.sign, state.overflow) = _arithmetic_condition_flags(
                            "sub", accumulator_value, destination_value, width_bits, 0
                        )
                        state.flags = ("cmp", _format_int(accumulator_value), _format_int(destination_value))
                        if equal:
                            result = _format_int(source_value)
                            assigned = self._assign(raw[0], result, state)
                            write_type = self._operand_type(raw[0], ins)
                        else:
                            result = _format_int(destination_value)
                            assigned = self._assign(accumulator or "rax", result, state)
                            write_type = self._operand_type(accumulator or "rax", ins)
                    else:
                        register_form = memory_form = False
                modeled = assigned is not None
                if not modeled:
                    # Preserve only proven single-thread value transfers. Unknown
                    # operands and unsupported addressing invalidate aliases; this
                    # does not model inter-thread atomic ordering or synchronization.
                    state.memory.clear()
                    state.memory_widths.clear()
                    state.memory_types.clear()
                    if any(_mem_inner(operand) is not None for operand in raw[:2]):
                        state.stack.clear()
                        state.stack_widths.clear()
                        state.stack_types.clear()
                    affected = raw[:2] if m == "xadd" else [raw[0], "rax"]
                    for operand in affected:
                        reg = _canon_reg(operand)
                        if reg:
                            state.regs[reg] = _UNKNOWN
                            _clear_register_slices(state, reg)
                            state.reg_types.pop(reg, None)
                    if memory_form:
                        self._assign(raw[0], _UNKNOWN, state)
                    state.flags = None
                    state.carry = state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
            elif m == "lea" and len(raw) == 2:
                if ins.reference_target is not None:
                    if ins.reference_target in self.strings:
                        val = json.dumps(self.strings[ins.reference_target], ensure_ascii=False)
                    else:
                        val = f"(void *)0x{ins.reference_target:X}"
                else:
                    address = raw[1].strip()
                    if address.startswith("[") and address.endswith("]"):
                        key = self._memory_key(address[1:-1], state)
                        if key.startswith("va:"):
                            val = f"(void *){key[3:]}"
                        elif self._pointer_expression(key) is not None:
                            val = key
                        else:
                            val = f"&({raw[1]})"
                    else:
                        val = f"&({raw[1]})"
                assigned = self._assign(raw[0], val, state)
                write_type = "uintptr_t"
            elif m in {"mul", "imul"} and len(raw) == 1:
                width_bits = _operand_width_bytes(raw[0], self.bits) * 8
                mask = (1 << width_bits) - 1
                accumulator = _parse_int(state.regs.get("rax", _UNKNOWN))
                operand_value = _parse_int(resolved[0])
                low_alias = {8: "ax", 16: "ax", 32: "eax", 64: "rax"}[width_bits]
                high_alias = {8: None, 16: "dx", 32: "edx", 64: "rdx"}[width_bits]

                def write_implicit_register(alias: str, value: int | None, part_width: int) -> str:
                    family = _canon_reg(alias)
                    assert family is not None
                    if value is None:
                        state.regs[family] = _UNKNOWN
                        _clear_register_slices(state, family)
                        state.reg_types.pop(family, None)
                        return _UNKNOWN
                    result_value = value & ((1 << part_width) - 1)
                    if part_width >= self.bits or (self.bits == 64 and part_width == 32):
                        full_value = result_value
                    else:
                        old_value = _parse_int(state.regs.get(family, _UNKNOWN))
                        if old_value is None:
                            state.regs[family] = _UNKNOWN
                            state.reg_types.pop(family, None)
                            return _UNKNOWN
                        full_mask = (1 << self.bits) - 1
                        low_mask = (1 << part_width) - 1
                        full_value = (old_value & (full_mask ^ low_mask)) | result_value
                    rendered = _format_int(full_value)
                    state.regs[family] = rendered
                    _clear_register_slices(state, family)
                    state.reg_types[family] = {8: "uint8_t", 16: "uint16_t", 32: "uint32_t", 64: "uint64_t"}[part_width]
                    return rendered

                if accumulator is None or operand_value is None:
                    low_value = high_value = None
                    carry_result = None
                else:
                    left = accumulator & mask
                    right = operand_value & mask
                    if m == "imul":
                        sign_bit = 1 << (width_bits - 1)
                        signed_left = left - (1 << width_bits) if left & sign_bit else left
                        signed_right = right - (1 << width_bits) if right & sign_bit else right
                        product = signed_left * signed_right
                        encoded = product & ((1 << (2 * width_bits)) - 1)
                        low_value, high_value = encoded & mask, (encoded >> width_bits) & mask
                        carry_result = not (-(1 << (width_bits - 1)) <= product < (1 << (width_bits - 1)))
                    else:
                        product = left * right
                        low_value, high_value = product & mask, (product >> width_bits) & mask
                        carry_result = high_value != 0
                low_output_width = 16 if width_bits == 8 else width_bits
                low_output_value = low_value if width_bits != 8 else (
                    None if low_value is None or high_value is None else low_value | (high_value << 8)
                )
                low_rendered = write_implicit_register(low_alias, low_output_value, low_output_width)
                assigned = ("rax", low_rendered)
                write_type = {8: "uint16_t", 16: "uint16_t", 32: "uint32_t", 64: "uint64_t"}[low_output_width]
                if width_bits != 8 and high_alias is not None:
                    high_rendered = write_implicit_register(high_alias, high_value, width_bits)
                    additional_writes.append({"target": _canon_reg(high_alias) or high_alias, "value": high_rendered})
                state.carry = carry_result
                state.overflow = carry_result
                state.zero = state.sign = state.parity = state.auxiliary = None
                state.flags = None
            elif m in {"bt", "bts", "btr", "btc"} and len(raw) >= 2:
                destination_reg = _canon_reg(raw[0])
                left_number = _parse_int(resolved[0])
                index_number = _parse_int(resolved[1])
                if destination_reg and left_number is not None and index_number is not None:
                    width_bits = _operand_width_bytes(raw[0], self.bits) * 8
                    bit_index = index_number % width_bits
                    bit_mask = 1 << bit_index
                    old_bit = bool(left_number & bit_mask)
                    result = left_number
                    if m == "bts":
                        result |= bit_mask
                    elif m == "btr":
                        result &= ~bit_mask
                    elif m == "btc":
                        result ^= bit_mask
                    if m != "bt":
                        rendered = _format_int(result & ((1 << width_bits) - 1))
                        assigned = self._assign(raw[0], rendered, state)
                        write_type = self._type_for_write(raw[0], rendered, state, ins)
                    state.carry = old_bit
                else:
                    # Memory bit indices can change the addressed word; without
                    # address-range tracking neither the target nor old bit is safe.
                    if m != "bt" and _mem_inner(raw[0]) is not None:
                        state.memory.clear()
                        state.memory_widths.clear()
                        state.memory_types.clear()
                        state.stack.clear()
                        state.stack_widths.clear()
                        state.stack_types.clear()
                    if m != "bt" and destination_reg:
                        state.regs[destination_reg] = _UNKNOWN
                        state.reg_types.pop(destination_reg, None)
                    state.carry = None
                state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
                state.flags = None
            elif re.fullmatch(r"v?cvt(?:ps2pd|pd2ps)", m) and len(raw) >= 2:
                is_vex = m.startswith("v")
                source_bits = _operand_width_bytes(raw[1], self.bits) * 8
                destination_bits = _operand_width_bytes(raw[0], self.bits) * 8
                value = _fold_packed_float_precision_convert(
                    m, resolved[1], source_bits, destination_bits
                )
                assigned = self._assign(raw[0], value, state, zero_upper_vector=is_vex)
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?(?:cvtdq2ps|cvttps2dq|cvtps2dq)", m) and len(raw) >= 2:
                is_vex = m.startswith("v")
                vector_bits = _operand_width_bytes(raw[0], self.bits) * 8
                value = _fold_packed_float_int_convert(m, resolved[1], vector_bits)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=is_vex)
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?(?:min|max)(?:ps|pd)", m) and len(raw) >= 2:
                is_vex = m.startswith("v")
                left = resolved[1] if is_vex and len(resolved) >= 3 else resolved[0]
                right = resolved[2] if is_vex and len(resolved) >= 3 else resolved[1]
                vector_bits = _operand_width_bytes(raw[0], self.bits) * 8
                value = _fold_packed_float_minmax(m, left, right, vector_bits)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=is_vex)
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?sqrt(?:ps|pd)", m) and len(raw) >= 2:
                is_vex = m.startswith("v")
                vector_bits = _operand_width_bytes(raw[0], self.bits) * 8
                value = _fold_packed_float_sqrt(m, resolved[1], vector_bits)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=is_vex)
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?cmpp[sd]", m) and len(raw) >= 3:
                is_vex = m.startswith("v")
                left = resolved[1] if is_vex and len(resolved) >= 4 else resolved[0]
                right = resolved[2] if is_vex and len(resolved) >= 4 else resolved[1]
                predicate = resolved[3] if is_vex and len(resolved) >= 4 else resolved[2]
                vector_bits = _operand_width_bytes(raw[0], self.bits) * 8
                value = _fold_packed_float_compare(m, left, right, predicate, vector_bits)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=is_vex)
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?(?:add|sub|mul|div)(?:ps|pd)", m) and len(raw) >= 2:
                is_vex = m.startswith("v")
                left = resolved[1] if is_vex and len(resolved) >= 3 else resolved[0]
                right = resolved[2] if is_vex and len(resolved) >= 3 else resolved[1]
                vector_bits = _operand_width_bytes(raw[0], self.bits) * 8
                value = _fold_packed_float(m, left, right, vector_bits)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=is_vex)
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?cvt(?:ss2sd|sd2ss)", m) and len(raw) >= (3 if m.startswith("v") else 2):
                is_vex = m.startswith("v")
                widening = m.endswith("ss2sd")
                source_width, target_width = (32, 64) if widening else (64, 32)
                base_vector = _parse_int(resolved[1] if is_vex else resolved[0])
                source_bits = _parse_int(resolved[-1])
                result_bits = None
                if base_vector is not None and source_bits is not None:
                    source_mask = (1 << source_width) - 1
                    source_fmt = "<f" if source_width == 32 else "<d"
                    target_fmt = "<d" if target_width == 64 else "<f"
                    source_value = struct.unpack(
                        source_fmt, (source_bits & source_mask).to_bytes(source_width // 8, "little")
                    )[0]
                    if not math.isnan(source_value):
                        try:
                            packed = struct.pack(target_fmt, source_value)
                            converted = struct.unpack(target_fmt, packed)[0]
                            if (math.isinf(source_value) and math.isinf(converted)
                                    and math.copysign(1.0, source_value) == math.copysign(1.0, converted)):
                                result_bits = int.from_bytes(packed, "little")
                            elif (math.isfinite(source_value) and math.isfinite(converted)
                                  and (source_value == 0.0 and converted == 0.0
                                       or Fraction.from_float(source_value) == Fraction.from_float(converted))
                                  and (source_value != 0.0
                                       or math.copysign(1.0, source_value) == math.copysign(1.0, converted))):
                                result_bits = int.from_bytes(packed, "little")
                        except (OverflowError, struct.error):
                            pass
                if result_bits is None:
                    value = _UNKNOWN
                else:
                    lane_mask = (1 << target_width) - 1
                    upper_mask = ((1 << 128) - 1) ^ lane_mask
                    value = _format_int((base_vector & upper_mask) | result_bits)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=is_vex)
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?round(?:ss|sd)", m) and len(raw) >= (4 if m.startswith("v") else 3):
                is_vex = m.startswith("v")
                float_width = 64 if m.endswith("sd") else 32
                lane_mask = (1 << float_width) - 1
                base_vector = _parse_int(resolved[1] if is_vex else resolved[0])
                source_bits = _parse_int(resolved[-2] if is_vex else resolved[1])
                immediate = _parse_int(resolved[-1])
                result_bits = None
                if base_vector is not None and source_bits is not None and immediate is not None:
                    raw_bits = source_bits & lane_mask
                    byte_count = float_width // 8
                    fmt = "<d" if float_width == 64 else "<f"
                    source_value = struct.unpack(fmt, raw_bits.to_bytes(byte_count, "little"))[0]
                    if not (immediate & 0x4) and math.isfinite(source_value) and not source_value.is_integer():
                        # MXCSR is not tracked, so a fractional source cannot be
                        # rounded precisely when the immediate selects MXCSR.
                        result_bits = None
                    elif not math.isfinite(source_value) or not (immediate & 0x4):
                        # NaNs/infinities are passed through; exactly integral
                        # finite values are invariant under all four modes.
                        result_bits = raw_bits
                    else:
                        mode = immediate & 0x3
                        if mode == 0:
                            rounded = round(source_value)
                        elif mode == 1:
                            rounded = math.floor(source_value)
                        elif mode == 2:
                            rounded = math.ceil(source_value)
                        else:
                            rounded = math.trunc(source_value)
                        rounded_float = (math.copysign(0.0, source_value) if rounded == 0
                                         else float(rounded))
                        try:
                            result_bits = int.from_bytes(
                                struct.pack(fmt, rounded_float), "little"
                            )
                        except (OverflowError, struct.error):
                            result_bits = None
                if result_bits is None:
                    value = _UNKNOWN
                else:
                    upper_mask = ((1 << 128) - 1) ^ lane_mask
                    value = _format_int((base_vector & upper_mask) | result_bits)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=is_vex)
                write_type = self._operand_type(raw[0], ins)
            elif m in {"ucomiss", "comiss", "ucomisd", "comisd"} and len(raw) >= 2:
                float_width = 64 if m.endswith("sd") else 32
                left_bits = _parse_int(resolved[0])
                right_bits = _parse_int(resolved[1])
                if left_bits is None or right_bits is None:
                    state.carry = state.zero = state.parity = None
                    state.auxiliary = state.sign = state.overflow = None
                    state.flags = None
                else:
                    byte_count = float_width // 8
                    fmt = "<d" if float_width == 64 else "<f"
                    mask = (1 << float_width) - 1
                    left = struct.unpack(fmt, (left_bits & mask).to_bytes(byte_count, "little"))[0]
                    right = struct.unpack(fmt, (right_bits & mask).to_bytes(byte_count, "little"))[0]
                    if math.isnan(left) or math.isnan(right):
                        state.carry = state.zero = state.parity = True
                    else:
                        state.carry = left < right
                        state.zero = left == right
                        state.parity = False
                    state.auxiliary = state.sign = state.overflow = False
                    state.flags = ("ucomi", _format_int(left_bits & mask), _format_int(right_bits & mask))
            elif re.fullmatch(r"v?(?:min|max)(?:ss|sd)", m) and len(raw) >= (3 if m.startswith("v") else 2):
                is_vex = m.startswith("v")
                float_width = 64 if m.endswith("sd") else 32
                lane_mask = (1 << float_width) - 1
                base_value = _parse_int(resolved[1] if is_vex else resolved[0])
                right_value = _parse_int(resolved[2] if is_vex else resolved[1])
                selected_bits = None
                if base_value is not None and right_value is not None:
                    left_bits, right_bits = base_value & lane_mask, right_value & lane_mask
                    fmt = "<d" if float_width == 64 else "<f"
                    byte_count = float_width // 8
                    left = struct.unpack(fmt, left_bits.to_bytes(byte_count, "little"))[0]
                    right = struct.unpack(fmt, right_bits.to_bytes(byte_count, "little"))[0]
                    # MIN/MAX return the second source for unordered or equal
                    # operands, preserving its NaN payload and signed-zero bits.
                    if math.isnan(left) or math.isnan(right) or left == right:
                        selected_bits = right_bits
                    elif (m[1:4] if is_vex else m[:3]) == "min":
                        selected_bits = left_bits if left < right else right_bits
                    else:
                        selected_bits = left_bits if left > right else right_bits
                if selected_bits is None:
                    value = _UNKNOWN
                else:
                    upper_mask = ((1 << 128) - 1) ^ lane_mask
                    value = _format_int((base_value & upper_mask) | selected_bits)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=is_vex)
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?(?:add|sub|mul|div)(?:ss|sd)", m) and len(raw) >= (3 if m.startswith("v") else 2):
                is_vex = m.startswith("v")
                float_width = 64 if m.endswith("sd") else 32
                lane_mask = (1 << float_width) - 1
                left_operand = resolved[1] if is_vex else resolved[0]
                right_operand = resolved[2] if is_vex else resolved[1]
                base_vector = _parse_int(left_operand)
                right_vector = _parse_int(right_operand)
                result_bits = None
                if base_vector is not None and right_vector is not None:
                    byte_count = float_width // 8
                    fmt = "<d" if float_width == 64 else "<f"
                    minimum_normal = 2.0 ** (-1022 if float_width == 64 else -126)
                    left_bits = base_vector & lane_mask
                    right_bits = right_vector & lane_mask
                    left = struct.unpack(fmt, left_bits.to_bytes(byte_count, "little"))[0]
                    right = struct.unpack(fmt, right_bits.to_bytes(byte_count, "little"))[0]
                    if (math.isfinite(left) and math.isfinite(right)
                            and abs(left) >= minimum_normal and abs(right) >= minimum_normal):
                        left_exact, right_exact = Fraction.from_float(left), Fraction.from_float(right)
                        operation = m[1:4] if is_vex else m[:3]
                        if operation == "add":
                            exact = left_exact + right_exact
                        elif operation == "sub":
                            exact = left_exact - right_exact
                        elif operation == "mul":
                            exact = left_exact * right_exact
                        elif right_exact:
                            exact = left_exact / right_exact
                        else:
                            exact = None
                        if exact:
                            try:
                                rounded = float(exact)
                                packed = struct.pack(fmt, rounded)
                                folded = struct.unpack(fmt, packed)[0]
                                if (math.isfinite(folded) and abs(folded) >= minimum_normal
                                        and Fraction.from_float(folded) == exact):
                                    result_bits = int.from_bytes(packed, "little")
                            except (OverflowError, struct.error):
                                pass
                if result_bits is None:
                    value = _UNKNOWN
                else:
                    upper_mask = ((1 << 128) - 1) ^ lane_mask
                    combined = (base_vector & upper_mask) | result_bits
                    value = _format_int(combined)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=is_vex)
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?cvtsi2(?:ss|sd)", m) and len(raw) >= (3 if m.startswith("v") else 2):
                is_vex = m.startswith("v")
                float_width = 64 if m.endswith("sd") else 32
                source_operand = raw[-1]
                source_value = _parse_int(resolved[-1])
                source_width = _operand_width_bytes(source_operand, self.bits) * 8
                destination_base = _parse_int(resolved[1] if is_vex else resolved[0])
                result_bits = None
                if source_value is not None and destination_base is not None and source_width > 0:
                    sign_bit = 1 << (source_width - 1)
                    signed_value = source_value & ((1 << source_width) - 1)
                    if signed_value & sign_bit:
                        signed_value -= 1 << source_width
                    fmt = "<d" if float_width == 64 else "<f"
                    try:
                        packed = struct.pack(fmt, float(signed_value))
                        converted = struct.unpack(fmt, packed)[0]
                        if math.isfinite(converted) and Fraction.from_float(converted) == signed_value:
                            result_bits = int.from_bytes(packed, "little")
                    except (OverflowError, struct.error):
                        pass
                if result_bits is None:
                    value = _UNKNOWN
                else:
                    lane_mask = (1 << float_width) - 1
                    upper_mask = ((1 << 128) - 1) ^ lane_mask
                    value = _format_int((destination_base & upper_mask) | result_bits)
                assigned = self._assign(raw[0], value, state, zero_upper_vector=is_vex)
                write_type = self._operand_type(raw[0], ins)
            elif re.fullmatch(r"v?cvt(?:t)?(?:ss|sd)2si", m) and len(raw) >= 2:
                float_width = 64 if "sd2si" in m else 32
                source_bits = _parse_int(resolved[1])
                destination_width = _operand_width_bytes(raw[0], self.bits) * 8
                if source_bits is None or destination_width <= 0:
                    value = _UNKNOWN
                else:
                    lane_mask = (1 << float_width) - 1
                    fmt = "<d" if float_width == 64 else "<f"
                    byte_count = float_width // 8
                    source = struct.unpack(fmt, (source_bits & lane_mask).to_bytes(byte_count, "little"))[0]
                    minimum_normal = 2.0 ** (-1022 if float_width == 64 else -126)
                    minimum = -(1 << (destination_width - 1))
                    maximum = (1 << (destination_width - 1)) - 1
                    if not math.isfinite(source) or (source != 0.0 and abs(source) < minimum_normal):
                        value = _UNKNOWN
                    elif "tt" in m:
                        integer = math.trunc(source)
                        value = _format_int(integer & ((1 << destination_width) - 1)) if minimum <= integer <= maximum else _UNKNOWN
                    elif source.is_integer():
                        integer = int(source)
                        value = _format_int(integer & ((1 << destination_width) - 1)) if minimum <= integer <= maximum else _UNKNOWN
                    else:
                        value = _UNKNOWN
                assigned = self._assign(raw[0], value, state)
                write_type = self._operand_type(raw[0], ins)
            elif m in {"div", "idiv"} and len(raw) == 1:
                width_bits = _operand_width_bytes(raw[0], self.bits) * 8
                mask = (1 << width_bits) - 1
                low_number = _parse_int(state.regs.get("rax", _UNKNOWN))
                divisor_number = _parse_int(resolved[0])
                high_number = _parse_int(state.regs.get("rdx", _UNKNOWN)) if width_bits > 8 else 0
                low_part = (low_number & (0xFFFF if width_bits == 8 else mask)) if low_number is not None else None
                high_part = (high_number & mask) if high_number is not None else None
                quotient: int | None = None
                remainder: int | None = None
                if low_part is not None and high_part is not None and divisor_number is not None:
                    dividend_bits = width_bits * 2 if width_bits > 8 else 16
                    dividend_mask = (1 << dividend_bits) - 1
                    dividend_raw = low_part if width_bits == 8 else ((high_part << width_bits) | low_part) & dividend_mask
                    divisor = divisor_number & mask
                    if m == "idiv":
                        dividend_sign = 1 << (dividend_bits - 1)
                        divisor_sign = 1 << (width_bits - 1)
                        dividend = dividend_raw - (1 << dividend_bits) if dividend_raw & dividend_sign else dividend_raw
                        signed_divisor = divisor - (1 << width_bits) if divisor & divisor_sign else divisor
                    else:
                        dividend, signed_divisor = dividend_raw, divisor
                    if signed_divisor != 0:
                        abs_quotient = abs(dividend) // abs(signed_divisor)
                        quotient_value = -abs_quotient if (dividend < 0) != (signed_divisor < 0) else abs_quotient
                        remainder_value = dividend - quotient_value * signed_divisor
                        if m == "idiv":
                            minimum, maximum = -(1 << (width_bits - 1)), (1 << (width_bits - 1)) - 1
                            fits = minimum <= quotient_value <= maximum
                        else:
                            fits = 0 <= quotient_value <= mask
                        if fits:
                            quotient, remainder = quotient_value & mask, remainder_value & mask

                def write_div_register(alias: str, value: int | None, part_width: int) -> str:
                    family = _canon_reg(alias)
                    assert family is not None
                    if value is None:
                        state.regs[family] = _UNKNOWN
                        _clear_register_slices(state, family)
                        state.reg_types.pop(family, None)
                        return _UNKNOWN
                    if part_width >= self.bits or (self.bits == 64 and part_width == 32):
                        full_value = value
                    else:
                        old_value = _parse_int(state.regs.get(family, _UNKNOWN))
                        if old_value is None:
                            state.regs[family] = _UNKNOWN
                            state.reg_types.pop(family, None)
                            return _UNKNOWN
                        full_mask = (1 << self.bits) - 1
                        low_mask = (1 << part_width) - 1
                        full_value = (old_value & (full_mask ^ low_mask)) | (value & low_mask)
                    rendered = _format_int(full_value)
                    state.regs[family] = rendered
                    _clear_register_slices(state, family)
                    state.reg_types[family] = {8: "uint8_t", 16: "uint16_t", 32: "uint32_t", 64: "uint64_t"}[part_width]
                    return rendered

                if width_bits == 8:
                    ax_value = None if quotient is None or remainder is None else quotient | (remainder << 8)
                    low_rendered = write_div_register("ax", ax_value, 16)
                    assigned = ("rax", low_rendered)
                    write_type = "uint8_t"
                else:
                    low_alias = {16: "ax", 32: "eax", 64: "rax"}[width_bits]
                    high_alias = {16: "dx", 32: "edx", 64: "rdx"}[width_bits]
                    low_rendered = write_div_register(low_alias, quotient, width_bits)
                    assigned = ("rax", low_rendered)
                    write_type = self._operand_type(low_alias, ins)
                    high_rendered = write_div_register(high_alias, remainder, width_bits)
                    additional_writes.append({"target": _canon_reg(high_alias) or high_alias, "value": high_rendered})
                state.carry = state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
                state.flags = None
            elif m in {"xor", "sub", "add", "adc", "sbb", "imul", "and", "or", "shl", "sal", "shr", "sar"} and len(raw) >= 2:
                prior_flag_state = (
                    state.carry, state.zero, state.sign, state.overflow,
                    state.parity, state.auxiliary, state.flags,
                )
                dst = raw[0]
                dst_reg = _canon_reg(dst)
                previous = self.resolve(dst, state, ins)
                width_bits = _operand_width_bytes(dst, self.bits) * 8
                lhs, rhs = previous, resolved[1]
                left_number, right_number = _parse_int(lhs), _parse_int(rhs)
                arithmetic_carry_in: int | None = 0
                if m in {"adc", "sbb"}:
                    carry_in = state.carry
                    arithmetic_carry_in = int(carry_in) if carry_in is not None else None
                    if left_number is not None and right_number is not None and carry_in is not None:
                        mask = (1 << width_bits) - 1
                        left_value = left_number & mask
                        right_value = right_number & mask
                        if m == "adc":
                            total = left_value + right_value + int(carry_in)
                            result = _format_int(total & mask)
                            state.carry = total > mask
                        else:
                            subtrahend = right_value + int(carry_in)
                            result = _format_int((left_value - subtrahend) & mask)
                            state.carry = left_value < subtrahend
                    else:
                        result = _UNKNOWN
                        state.carry = None
                elif m == "xor" and raw[0].strip().lower() == raw[1].strip().lower():
                    result = "0"
                    state.carry = False
                else:
                    op = _BINARY_OPS.get(m)
                    if len(raw) >= 3:
                        lhs, rhs = resolved[1], resolved[2]
                    else:
                        lhs, rhs = previous, resolved[1]
                    left_number, right_number = _parse_int(lhs), _parse_int(rhs)
                    result = (
                        _fold_alu(m, op, lhs, rhs, width_bits)
                        if op else _UNKNOWN
                    )
                    if m in {"add", "sub"} and left_number is not None and right_number is not None:
                        mask = (1 << width_bits) - 1
                        left_value, right_value = left_number & mask, right_number & mask
                        state.carry = (left_value + right_value > mask) if m == "add" else left_value < right_value
                    elif m in {"and", "or", "xor"}:
                        state.carry = False
                    else:
                        state.carry = None
                if m in {"add", "sub", "adc", "sbb"}:
                    if left_number is not None and right_number is not None and arithmetic_carry_in is not None:
                        (_result, state.carry, state.parity, state.auxiliary, state.zero,
                         state.sign, state.overflow) = _arithmetic_condition_flags(
                            m, left_number, right_number, width_bits, arithmetic_carry_in
                        )
                    else:
                        state.carry = state.parity = state.auxiliary = state.zero = None
                        state.sign = state.overflow = None
                elif m in {"and", "or", "xor"}:
                    result_number = _parse_int(result)
                    state.carry = False
                    state.overflow = False
                    state.auxiliary = None
                    if result_number is None:
                        state.parity = state.zero = state.sign = None
                    else:
                        state.zero, state.sign, state.parity = _result_condition_flags(result_number, width_bits)
                elif m == "imul":
                    if left_number is None or right_number is None:
                        state.carry = state.overflow = None
                    else:
                        sign_bit = 1 << (width_bits - 1)
                        signed_left = (left_number & ((1 << width_bits) - 1))
                        signed_right = (right_number & ((1 << width_bits) - 1))
                        if signed_left & sign_bit:
                            signed_left -= 1 << width_bits
                        if signed_right & sign_bit:
                            signed_right -= 1 << width_bits
                        product = signed_left * signed_right
                        overflow = not (-(1 << (width_bits - 1)) <= product < (1 << (width_bits - 1)))
                        state.carry = state.overflow = overflow
                    state.zero = state.sign = state.parity = state.auxiliary = None
                elif m in {"shl", "sal", "shr", "sar"}:
                    count = None if right_number is None else right_number & (0x3F if width_bits == 64 else 0x1F)
                    original = None if left_number is None else left_number & ((1 << width_bits) - 1)
                    shifted = _parse_int(result)
                    if count == 0:
                        # A zero effective count leaves all flags untouched.
                        (state.carry, state.zero, state.sign, state.overflow,
                         state.parity, state.auxiliary, state.flags) = prior_flag_state
                    elif original is None or shifted is None or count is None:
                        state.carry = state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
                    else:
                        if count <= width_bits:
                            if m in {"shl", "sal"}:
                                state.carry = bool((original >> (width_bits - count)) & 1)
                            else:
                                state.carry = bool((original >> (count - 1)) & 1)
                        else:
                            state.carry = None
                        state.zero, state.sign, state.parity = _result_condition_flags(shifted, width_bits)
                        state.auxiliary = None
                        if count != 1:
                            state.overflow = None
                        elif m in {"shl", "sal"}:
                            state.overflow = bool((shifted >> (width_bits - 1)) & 1) != bool(state.carry)
                        elif m == "shr":
                            state.overflow = bool((original >> (width_bits - 1)) & 1)
                        else:
                            state.overflow = False
                else:
                    state.carry = state.parity = state.auxiliary = state.zero = None
                    state.sign = state.overflow = None
                assigned = self._assign(dst, result, state, raw[1] if len(raw) > 1 else None)
                write_type = self._operand_type(dst, ins)
                if m not in {"shl", "sal", "shr", "sar"} or right_number is None or (right_number & (0x3F if width_bits == 64 else 0x1F)) != 0:
                    state.flags = (m, result, "0")
            elif m in {"inc", "dec"} and raw:
                old = self.resolve(raw[0], state, ins)
                width_bits = _operand_width_bytes(raw[0], self.bits) * 8
                result = _fold_unary_alu(m, old, width_bits)
                assigned = self._assign(raw[0], result, state)
                write_type = self._operand_type(raw[0], ins)
                state.flags = (m, result, "0")
                old_number = _parse_int(old)
                if old_number is None:
                    state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
                else:
                    mask = (1 << width_bits) - 1
                    old_value = old_number & mask
                    result_number = (old_value + (1 if m == "inc" else -1)) & mask
                    state.zero, state.sign, state.parity = _result_condition_flags(result_number, width_bits)
                    sign_bit = 1 << (width_bits - 1)
                    state.overflow = old_value == (sign_bit - 1) if m == "inc" else old_value == sign_bit
                    state.auxiliary = (old_value & 0xF) == 0xF if m == "inc" else (old_value & 0xF) == 0
            elif m == "neg" and raw:
                old = self.resolve(raw[0], state, ins)
                width_bits = _operand_width_bytes(raw[0], self.bits) * 8
                result = _fold_unary_alu(m, old, width_bits)
                assigned = self._assign(raw[0], result if len(result) <= 160 else _UNKNOWN, state)
                write_type = self._operand_type(raw[0], ins)
                state.flags = (m, result, "0")
                old_number = _parse_int(old)
                if old_number is None:
                    state.carry = state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
                else:
                    mask = (1 << width_bits) - 1
                    old_value = old_number & mask
                    result_number = (-old_value) & mask
                    state.carry = old_value != 0
                    state.zero, state.sign, state.parity = _result_condition_flags(result_number, width_bits)
                    state.overflow = old_value == (1 << (width_bits - 1))
                    state.auxiliary = (old_value & 0xF) != 0
            elif m in {"cmp", "test"} and len(resolved) >= 1:
                left = resolved[0]
                right = resolved[1] if len(resolved) > 1 else resolved[0]
                state.flags = (m, left, right)
                if m == "test":
                    state.carry = False
                    state.overflow = False
                    state.auxiliary = None
                    left_number, right_number = _parse_int(left), _parse_int(right)
                    if left_number is None or right_number is None:
                        state.zero = state.sign = state.parity = None
                    else:
                        width_bits = _operand_width_bytes(raw[0], self.bits) * 8
                        mask = (1 << width_bits) - 1
                        result = (left_number & right_number) & mask
                        state.zero, state.sign, state.parity = _result_condition_flags(result, width_bits)
                else:
                    left_number, right_number = _parse_int(left), _parse_int(right)
                    if left_number is None or right_number is None:
                        state.carry = state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
                    else:
                        width_bits = _operand_width_bytes(raw[0], self.bits) * 8
                        (_result, state.carry, state.parity, state.auxiliary, state.zero,
                         state.sign, state.overflow) = _arithmetic_condition_flags(
                            "cmp", left_number, right_number, width_bits
                        )
            elif m == "clc":
                state.carry = False
            elif m == "stc":
                state.carry = True
            elif m == "cmc":
                state.carry = None if state.carry is None else not state.carry
            elif ins.flow == "call":
                if self.bits == 64:
                    # Preserve the pre-call Windows x64 register arguments for
                    # callback/function-pointer discovery before volatile clobbering.
                    call_argument_values = [
                        state.regs.get(reg, _UNKNOWN) for reg in ("rcx", "rdx", "r8", "r9")
                    ]
                else:
                    # Same-block push tracking supplies a bounded x86 stack-argument
                    # view; disagreement at CFG joins clears it in AbstractState.merge.
                    call_argument_values = list(reversed(state.pending_pushes[-16:]))
                    # Also expose must-known values explicitly written into the
                    # current ESP-relative outgoing argument area. Keep this
                    # separate from push order because mixed push/store layouts
                    # cannot be ordered safely without a complete stack model.
                    stack_slots = []
                    for key, value in state.stack.items():
                        address = _stack_address(key)
                        if (address is None or address[0] != "rsp" or address[1] < 0
                                or state.stack_widths.get(key) != 4 or value == _UNKNOWN):
                            continue
                        stack_slots.append((address[1], value))
                    call_stack_memory_values = [
                        value for _offset, value in sorted(stack_slots)[:16]
                    ]
                # Resolve a function pointer when value propagation proves that the
                # call operand is a known code address.  Keep the instruction's
                # original direct_target untouched: the JSON row records that this
                # was an indirect call and separately exposes the proven target.
                if ins.direct_target is not None:
                    resolved_call_target = ins.direct_target
                elif resolved:
                    target_text = resolved[0].strip()
                    import_match = re.fullmatch(r"iat\((.+)\)", target_text, re.I)
                    if import_match:
                        resolved_call_import = import_match.group(1)
                    static_match = re.fullmatch(r"static_ptr\((0x[0-9a-f]+|\d+)\)", target_text, re.I)
                    if static_match:
                        target_text = static_match.group(1)
                        resolved_call_provenance = "static_pointer_slot"
                    address_match = re.fullmatch(
                        r"(?:\(void \*\))?\s*(0x[0-9a-f]+|[0-9]+)",
                        target_text,
                        re.I,
                    )
                    if address_match:
                        try:
                            candidate_target = int(address_match.group(1), 0)
                            if candidate_target in self.known_code_addresses:
                                resolved_call_target = candidate_target
                        except ValueError:
                            resolved_call_target = None
                if ins.import_name:
                    name = ins.import_name.split("!")[-1]
                    resolved_call_import = ins.import_name
                elif resolved_call_import is not None:
                    name = resolved_call_import.split("!")[-1]
                elif resolved_call_target is not None:
                    name = self.function_names.get(
                        resolved_call_target, f"sub_{resolved_call_target:X}"
                    )
                elif ins.reference_target in self.imports:
                    name = self.imports[ins.reference_target].split("!")[-1]
                else:
                    name = "indirect_call"
                call_argument_types = list(API_PARAMETER_TYPES.get(
                    normalize_api_name(resolved_call_import or ins.import_name or name), ()
                ))
                args = []
                args_known = False
                if resolved_call_target is not None and self.bits == 64:
                    arity = self.function_arg_counts.get(resolved_call_target)
                    abi_regs = ("rcx", "rdx", "r8", "r9")
                    if arity is not None:
                        args_known = True
                        args = [state.regs.get(reg, f"{reg}_unknown") for reg in abi_regs[:arity]]
                elif resolved_call_target is not None and self.bits == 32:
                    # For x86, the bounded push tracker is already in caller
                    # argument order. Use it only when a proven callee target
                    # has a known arity; otherwise keep the call opaque.
                    arity = self.function_arg_counts.get(resolved_call_target)
                    if arity is not None:
                        args_known = True
                        visible = call_argument_values or []
                        args = list(visible[:arity])
                        args.extend(
                            f"unknown_stack_arg_{index}"
                            for index in range(len(args) + 1, arity + 1)
                        )
                call_expr = f"{name}({', '.join(args)})" if args_known else f"{name}(/* arguments unresolved */)"
                if len(call_expr) > 160:
                    call_expr = _UNKNOWN
                # Windows x64 volatile integer registers are clobbered by calls.
                for reg in ("rcx", "rdx", "r8", "r9", "r10", "r11"):
                    state.regs.pop(reg, None)
                    _clear_register_slices(state, reg)
                # A call may mutate arbitrary memory. Stack slots remain tracked
                # because their addresses are frame-relative and separately modeled.
                state.memory.clear()
                state.memory_widths.clear()
                state.memory_types.clear()
                write_type = self._call_return_type(ins, name, resolved_call_target)
                if write_type == "void":
                    state.regs.pop("rax", None)
                    state.reg_types.pop("rax", None)
                else:
                    state.regs["rax"] = call_expr
                    state.reg_types["rax"] = write_type
                    assigned = ("rax", call_expr)
                _clear_register_slices(state, "rax")
                state.flags = None
                state.carry = None
                state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
                state.direction = None
                state.pending_pushes.clear()
                state.x87_stack = None
            elif m in {"loop", "loope", "loopz", "loopne", "loopnz"}:
                count_alias = "rcx" if self.bits == 64 else "ecx"
                count = _parse_int(self.resolve(count_alias, state, ins))
                if count is None:
                    assigned = self._assign(count_alias, _UNKNOWN, state)
                else:
                    width = self.bits
                    decremented = (count - 1) & ((1 << width) - 1)
                    assigned = self._assign(count_alias, _format_int(decremented), state)
                additional_writes.append({"target": count_alias, "value": assigned[1] if assigned else _UNKNOWN})
                condition = self._branch_condition(m, state)
            elif ins.flow == "conditional":
                condition = self._branch_condition(m, state)
            elif m.startswith("ret"):
                pass
            elif ins.flow == "jump" or m in {"nop", "endbr32", "endbr64", "int3", "ud2"}:
                pass
            elif m in {"lfence", "sfence", "mfence", "serialize", "pause"}:
                pass
            elif m in {"push", "pop", "leave"}:
                # Stack manipulation is not part of the current value model.
                # Drop stack and RSP facts rather than carrying stale slots.
                state.stack.clear()
                state.stack_widths.clear()
                state.stack_types.clear()
                state.regs.pop("rsp", None)
                state.reg_types.pop("rsp", None)
                _clear_register_slices(state, "rsp")
                if m == "push" and raw:
                    state.pending_pushes.append(resolved[0])
                    if len(state.pending_pushes) > 16:
                        state.pending_pushes.pop(0)
                elif m == "pop":
                    if state.pending_pushes:
                        state.pending_pushes.pop()
                    if raw:
                        destination_reg = _canon_reg(raw[0])
                        if destination_reg:
                            _clear_register_slices(state, destination_reg)
                            state.regs[destination_reg] = _UNKNOWN
                            state.reg_types.pop(destination_reg, None)
                else:
                    state.pending_pushes.clear()
            else:
                # Consume the normalized machine effects so the value-flow pass
                # invalidates the registers/memory the lifter identifies. Opaque
                # effects remain conservative and are never treated as no-ops.
                semantic_effects = lift_instruction(ins, bits=self.bits).get("semantics", {})
                for destination_reg in semantic_effects.get("register_writes", ()):
                    if destination_reg in state.regs or destination_reg in {
                        "RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "RBP", "RSP",
                        "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15",
                    }:
                        _clear_register_slices(state, destination_reg)
                        state.regs[destination_reg] = _UNKNOWN
                        state.reg_types.pop(destination_reg, None)
                        additional_writes.append({"target": destination_reg, "value": _UNKNOWN})
                memory_effects = semantic_effects.get("memory_effects", {})
                if memory_effects.get("writes") or memory_effects.get("unknown"):
                    state.memory.clear()
                    state.memory_widths.clear()
                    state.memory_types.clear()
                    state.stack.clear()
                    state.stack_widths.clear()
                    state.stack_types.clear()
                    additional_writes.append({"target": "memory[*]", "value": _UNKNOWN})
                if not raw and not semantic_effects.get("register_writes"):
                    state.regs.clear()
                    state.reg_types.clear()
                    state.reg_slices.clear()
                    state.memory.clear()
                    state.memory_widths.clear()
                    state.memory_types.clear()
                    state.stack.clear()
                    state.stack_widths.clear()
                    state.stack_types.clear()
                    additional_writes.append({"target": "state[*]", "value": _UNKNOWN})
                flag_effects = semantic_effects.get("flag_effects", {})
                if flag_effects.get("unknown") or flag_effects.get("writes") or flag_effects.get("undefined"):
                    state.flags = None
                    state.carry = None
                    state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
                if m.startswith("f"):
                    state.x87_stack = None

            if self.bits == 32 and m == "add" and len(raw) >= 2 and _canon_reg(raw[0]) == "rsp":
                cleanup_bytes = _parse_int(resolved[1])
                if cleanup_bytes is not None and cleanup_bytes > 0:
                    cleanup_count = min(len(state.pending_pushes), cleanup_bytes // 4)
                    if cleanup_count:
                        del state.pending_pushes[-cleanup_count:]

            if m in {
                "div", "idiv",
            }:
                state.carry = None
                state.flags = None
                state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
            elif m in {"popf", "popfd", "popfq", "sahf", "iret", "iretd", "iretq"}:
                state.carry = None
                state.flags = None
                state.zero = state.sign = state.overflow = state.parity = state.auxiliary = None
                if m in {"popf", "popfd", "popfq", "iret", "iretd", "iretq"}:
                    state.direction = None

            if assigned and m.startswith("v") and raw:
                vector_destination = re.fullmatch(r"([xyz])mm([0-9]|[12][0-9]|3[01])", raw[0].strip().lower())
                if vector_destination:
                    prefix, number = vector_destination.groups()
                    additional_writes.extend(
                        {"target": f"{alias}mm{number}", "value": state.regs[f"{alias}mm{number}"]}
                        for alias in ("x", "y", "z")
                        if alias != prefix and f"{alias}mm{number}" in state.regs
                    )

            if assigned and write_type and raw and ins.flow != "call":
                self._set_type(raw[0], write_type, state)
            if collect:
                # Record every decoded instruction within budget; this report is intended
                # to be machine-readable evidence for future higher-level reconstruction.
                row = {
                    "function_entry": hex(function_entry), "block": hex(block.start),
                    "address": hex(ins.address), "instruction": ins.text,
                    "mnemonic": m, "prefixes": prefixes,
                    "memory_ordering_fence": {
                        "lfence": "load_and_execution", "sfence": "store",
                        "mfence": "full", "serialize": "serializing",
                    }.get(m),
                    "resolved_operands": resolved, "write": {"target": assigned[0], "value": assigned[1]} if assigned else None,
                    "additional_writes": additional_writes,
                    "semantic_effects": lift_instruction(ins, bits=self.bits).get("semantics", {}),
                    "direction_flag_after": state.direction,
                    "resolved_call_target": (
                        hex(resolved_call_target)
                        if ins.flow == "call" and resolved_call_target is not None
                        else None
                    ),
                    "resolved_indirect_jump_target": (
                        hex(resolved_indirect_jump_target)
                        if resolved_indirect_jump_target is not None else None
                    ),
                    "resolved_call_import": resolved_call_import,
                    "call_argument_values": call_argument_values,
                    "call_argument_types": call_argument_types,
                    "call_stack_memory_values": call_stack_memory_values,
                    "resolved_call_provenance": resolved_call_provenance,
                    "condition": condition,
                    "write_type": write_type,
                    "registers_after": dict(sorted(state.regs.items())),
                    "register_slices_after": dict(sorted(state.reg_slices.items())),
                    "x87_stack_after": list(state.x87_stack) if state.x87_stack is not None else None,
                    "stack_slots_after": dict(sorted(state.stack.items())),
                    "stack_slot_widths_after": dict(sorted(state.stack_widths.items())),
                    "memory_after": dict(sorted(state.memory.items())),
                    "memory_widths_after": dict(sorted(state.memory_widths.items())),
                    "memory_types_after": dict(sorted(state.memory_types.items())),
                    "register_types_after": dict(sorted(state.reg_types.items())),
                    "stack_slot_types_after": dict(sorted(state.stack_types.items())),
                    "carry_flag_after": state.carry,
                    "zero_flag_after": state.zero,
                    "sign_flag_after": state.sign,
                    "overflow_flag_after": state.overflow,
                    "parity_flag_after": state.parity,
                    "auxiliary_flag_after": state.auxiliary,
                }
                rows.append(row)
        return state, rows

    def analyze(self, entry: int, blocks: list[BasicBlock], max_rows: int = 100_000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.steps = 0
        self.warnings = []
        block_by_start = {b.start: b for b in blocks}
        if entry not in block_by_start:
            return [], {"entry": hex(entry), "blocks_analyzed": 0, "rows": 0, "warnings": ["Entry block not available."]}
        allowed = set(block_by_start)
        in_states: dict[int, AbstractState] = {entry: self.initial_state()}
        queue = deque([entry])
        queued = {entry}
        visits: dict[int, int] = {}
        iterations = 0
        iteration_cap = max(100, min(self.max_steps_per_function, len(blocks) * self.max_visits_per_block * 4))
        while queue and iterations < iteration_cap and self.steps < self.max_steps_per_function:
            cur = queue.popleft(); queued.discard(cur)
            if cur not in block_by_start or cur not in in_states:
                continue
            iterations += 1
            visits[cur] = visits.get(cur, 0) + 1
            out_state, _ = self.transfer(block_by_start[cur], in_states[cur], entry, collect=False)
            successors = list(block_by_start[cur].successors)
            terminal_branch = (
                block_by_start[cur].instructions[-1]
                if block_by_start[cur].instructions
                and block_by_start[cur].instructions[-1].flow == "conditional"
                else None
            )
            branch_target = terminal_branch.direct_target if terminal_branch else None
            if terminal_branch and branch_target in successors:
                branch_parts = terminal_branch.mnemonic.lower().strip().split()
                branch_mnemonic = branch_parts[-1] if branch_parts else ""
                branch_result = self._branch_condition(branch_mnemonic, out_state)
                if branch_result == "1":
                    successors = [branch_target]
                elif branch_result == "0":
                    successors = [succ for succ in successors if succ != branch_target]
            for succ in successors:
                if succ not in allowed:
                    continue
                if succ not in in_states:
                    in_states[succ] = out_state.clone()
                    if succ not in queued:
                        queue.append(succ); queued.add(succ)
                    continue
                merged = AbstractState.merge(in_states[succ], out_state)
                if merged != in_states[succ]:
                    in_states[succ] = merged
                    if succ not in queued:
                        queue.append(succ); queued.add(succ)
        if queue:
            self.warnings.append(f"Data-flow fixed-point limit reached for function {entry:#x}; results are partial.")
        if self.steps >= self.max_steps_per_function:
            self.warnings.append(f"Data-flow instruction limit reached for function {entry:#x}; results are partial.")
        rows: list[dict[str, Any]] = []
        for block_start in sorted(in_states):
            if len(rows) >= max_rows:
                self.warnings.append(f"Value-flow row limit reached ({max_rows}); function output is partial.")
                break
            _, block_rows = self.transfer(block_by_start[block_start], in_states[block_start], entry, collect=True)
            take = max_rows - len(rows)
            rows.extend(block_rows[:take])
        summary = {
            "entry": hex(entry), "blocks_reachable": len(in_states), "analysis_steps": self.steps,
            "rows": len(rows), "known_register_values_at_block_entries": sum(len(s.regs) for s in in_states.values()),
            "known_stack_values_at_block_entries": sum(len(s.stack) for s in in_states.values()),
            "warnings": list(dict.fromkeys(self.warnings)),
        }
        return rows, summary
