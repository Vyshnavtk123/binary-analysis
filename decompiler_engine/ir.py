from __future__ import annotations

"""Small independent x86/x64 instruction-to-IR lifter.

This is a research IR, not a complete x86 semantic specification. Operations that
cannot be modeled safely are preserved as opaque operations with their original text.
"""

import re
from typing import Any

from .models import BasicBlock, Instruction

_REG_ALIASES: dict[str, str] = {}
def _aliases(family: str, *names: str) -> None:
    for name in names:
        _REG_ALIASES[name] = family

_aliases("RAX", "rax", "eax", "ax", "al", "ah")
_aliases("RBX", "rbx", "ebx", "bx", "bl", "bh")
_aliases("RCX", "rcx", "ecx", "cx", "cl", "ch")
_aliases("RDX", "rdx", "edx", "dx", "dl", "dh")
_aliases("RSI", "rsi", "esi", "si", "sil")
_aliases("RDI", "rdi", "edi", "di", "dil")
_aliases("RBP", "rbp", "ebp", "bp", "bpl")
_aliases("RSP", "rsp", "esp", "sp", "spl")
_aliases("RIP", "rip", "eip", "ip")
for _i in range(8, 16):
    _aliases(f"R{_i}", f"r{_i}", f"r{_i}d", f"r{_i}w", f"r{_i}b")
for _i in range(32):
    _aliases(f"XMM{_i}", f"xmm{_i}")
    _aliases(f"YMM{_i}", f"ymm{_i}")
    _aliases(f"ZMM{_i}", f"zmm{_i}")

_REG_WIDTH = {
    **{n: 8 for n in ("al", "ah", "bl", "bh", "cl", "ch", "dl", "dh", "sil", "dil", "bpl", "spl")},
    **{n: 16 for n in ("ax", "bx", "cx", "dx", "si", "di", "bp", "sp")},
    **{n: 32 for n in ("eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp", "eip")},
    **{f"r{i}d": 32 for i in range(8, 16)},
    **{f"r{i}w": 16 for i in range(8, 16)},
    **{f"r{i}b": 8 for i in range(8, 16)},
    **{f"xmm{i}": 128 for i in range(32)},
    **{f"ymm{i}": 256 for i in range(32)},
    **{f"zmm{i}": 512 for i in range(32)},
}

_BINARY = {
    "add": "+", "sub": "-", "imul": "*", "mul": "*", "and": "&",
    "or": "|", "xor": "^", "shl": "<<", "sal": "<<", "shr": ">>", "sar": ">>",
    "sbb": "-", "adc": "+",
}
_UNARY = {"inc": "+1", "dec": "-1", "neg": "neg", "not": "not"}
_COND = {
    "je", "jz", "jne", "jnz", "ja", "jae", "jb", "jbe", "jg", "jge", "jl", "jle",
    "jo", "jno", "js", "jns", "jp", "jnp", "jpe", "jpo", "jecxz", "jrcxz", "jcxz",
    "loop", "loope", "loopz", "loopne", "loopnz", "jnae", "jnb", "jnc", "jc", "jna", "jnbe", "jnl", "jnle", "jnge", "jng",
}
_INSTRUCTION_PREFIXES = {"lock", "rep", "repe", "repz", "repne", "repnz"}
_STRING_INSTRUCTIONS = {
    **{f"{stem}{suffix}": (kind, width) for stem, kind in (
        ("movs", "move"), ("cmps", "compare"), ("stos", "store"),
        ("lods", "load"), ("scas", "scan"), ("ins", "input"), ("outs", "output"),
    ) for suffix, width in (("b", 8), ("w", 16), ("d", 32))},
    "movsq": ("move", 64), "cmpsq": ("compare", 64), "stosq": ("store", 64),
    "lodsq": ("load", 64), "scasq": ("scan", 64),
}


def split_operands(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for char in text:
        if char == "[": depth += 1
        elif char == "]": depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf.clear()
        else:
            buf.append(char)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _integer(text: str) -> int | None:
    s = text.strip().lower().replace("_", "")
    if re.fullmatch(r"[-+]?0x[0-9a-f]+", s):
        try: return int(s, 16)
        except ValueError: return None
    if re.fullmatch(r"[-+]?\d+", s):
        try: return int(s, 10)
        except ValueError: return None
    if re.fullmatch(r"[0-9a-f]+h", s):
        try: return int(s[:-1], 16)
        except ValueError: return None
    return None


def _family(token: str) -> str | None:
    return _REG_ALIASES.get(token.strip().lower())


def raw_destination_is_32bit(token: str) -> bool:
    low = token.strip().lower()
    return low in {"eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp", "eip"} or bool(re.fullmatch(r"r(?:[89]|1[0-5])d", low))


def _stack_name(inner: str) -> str | None:
    compact = re.sub(r"\s+", "", inner.lower())
    match = re.fullmatch(r"(rbp|ebp|rsp|esp)([+-](?:0x[0-9a-f]+|\d+))?", compact)
    if not match:
        return None
    base = "RBP" if match.group(1) in {"rbp", "ebp"} else "RSP"
    disp = match.group(2) or "+0"
    val = _integer(disp)
    if val is None:
        return None
    sign = "P" if val >= 0 else "M"
    return f"STACK_{base}_{sign}{abs(val):X}"


def normalize_operand(operand: str, ins: Instruction | None = None, destination: bool = False) -> dict[str, Any]:
    """Normalize an Intel-syntax operand into a typed symbolic token."""
    raw = operand.strip()
    low = raw.lower()
    reg = _family(low)
    if reg:
        return {"text": raw, "kind": "var", "value": reg, "width": _REG_WIDTH.get(low, 64 if reg.startswith("R") else 32)}
    if low.startswith("byte ptr "): width = 8
    elif low.startswith("word ptr "): width = 16
    elif low.startswith("dword ptr "): width = 32
    elif low.startswith("qword ptr "): width = 64
    elif low.startswith("xmmword ptr "): width = 128
    elif low.startswith("ymmword ptr "): width = 256
    elif low.startswith("zmmword ptr "): width = 512
    else: width = None
    mem = re.match(r"\s*(?:(?:byte|word|dword|qword|xmmword|ymmword|zmmword)\s+ptr\s+)?\[(.+)\]\s*$", raw, re.I)
    if mem:
        inner = re.sub(r"\s+", " ", mem.group(1).strip().lower())
        stack = _stack_name(inner)
        if stack:
            return {"text": raw, "kind": "var", "value": stack, "width": width, "storage": "stack"}
        # Replace register aliases inside a memory expression without inventing pointer semantics.
        def repl(match: re.Match[str]) -> str:
            return _family(match.group(0)) or match.group(0).upper()
        inner_norm = re.sub(r"(?<![A-Za-z0-9_])(?:r(?:1[0-5]|[89])(?:[dwb])?|[er]?(?:ax|bx|cx|dx|si|di|bp|sp|ip)|[abcd][lh])(?![A-Za-z0-9_])", repl, inner, flags=re.I)
        if ins is not None and "rip" in inner and ins.reference_target is not None:
            return {"text": raw, "kind": "address", "value": hex(ins.reference_target), "width": width}
        return {"text": raw, "kind": "memory", "value": f"MEM[{inner_norm}]", "width": width}
    number = _integer(low)
    if number is not None:
        return {"text": raw, "kind": "const", "value": number, "width": width}
    if low in {"eflags", "rflags", "flags"}:
        return {"text": raw, "kind": "flags", "value": "FLAGS", "width": None}
    return {"text": raw, "kind": "opaque", "value": raw, "width": width}


def _value(token: dict[str, Any]) -> str:
    kind, value = token["kind"], token["value"]
    if kind == "const": return f"const:{value}"
    if kind == "address": return f"address:{value}"
    if kind == "var": return str(value)
    if kind == "memory": return str(value)
    if kind == "flags": return "FLAGS"
    return f"opaque:{value}"


def _type_for_token(token: dict[str, Any], bits: int = 64) -> str:
    """Return a conservative C-like type hint for a normalized operand."""
    explicit = token.get("type")
    if isinstance(explicit, str):
        return explicit
    kind = token.get("kind")
    width = token.get("width")
    if kind == "address":
        return "uintptr_t"
    if kind == "flags":
        return "flags_t"
    if kind == "memory":
        return {
            8: "uint8_t", 16: "uint16_t", 32: "uint32_t",
            64: "uint64_t", 128: "vector128_t", 256: "vector256_t",
            512: "vector512_t",
        }.get(width, "unknown")
    if kind == "const":
        return {
            8: "uint8_t", 16: "uint16_t", 32: "uint32_t",
            64: "uint64_t",
        }.get(width, "int")
    if kind == "var":
        value = str(token.get("value", ""))
        if value.startswith("ST"):
            return "long double"
        if value.startswith("STACK_"):
            return {
                8: "uint8_t", 16: "uint16_t", 32: "uint32_t",
                64: "uint64_t",
            }.get(width, "stack_value_t")
        if value.startswith("XMM"):
            return "vector128_t"
        if value.startswith("YMM"):
            return "vector256_t"
        if value.startswith("ZMM"):
            return "vector512_t"
        if value in {"RSP", "RBP", "RIP"} or value.startswith("R"):
            return {
                32: "uint32_t", 64: "uint64_t",
            }.get(width or bits, "uintptr_t")
    return "unknown"


def _type_for_destination(destination: str | None, width: int | None, bits: int = 64) -> str | None:
    if destination == "X87_STACK" or (destination and destination.startswith("ST")):
        return "long double"
    if destination == "FLAGS":
        return "flags_t"
    if destination is None:
        return None
    if destination.startswith("STACK_"):
        return {
            8: "uint8_t", 16: "uint16_t", 32: "uint32_t",
            64: "uint64_t",
        }.get(width, "stack_value_t")
    if destination.startswith("XMM"):
        return "vector128_t"
    if destination.startswith("YMM"):
        return "vector256_t"
    if destination.startswith("ZMM"):
        return "vector512_t"
    if destination in {"RSP", "RBP", "RIP"}:
        return "uintptr_t"
    return {
        8: "uint8_t", 16: "uint16_t", 32: "uint32_t",
        64: "uint64_t",
    }.get(width or bits, "unknown")


def _lift_instruction_core(ins: Instruction, bits: int = 64, block_start: int | None = None) -> dict[str, Any]:
    """Map one assembly instruction to a normalized, analysis-friendly IR record."""
    full_mnemonic = ins.mnemonic.lower().strip()
    mnemonic_parts = full_mnemonic.split()
    prefixes: list[str] = []
    while mnemonic_parts and mnemonic_parts[0] in _INSTRUCTION_PREFIXES:
        prefixes.append(mnemonic_parts.pop(0))
    m = mnemonic_parts[-1] if mnemonic_parts else full_mnemonic
    operands = split_operands(ins.operands)
    tokens = [normalize_operand(x, ins) for x in operands]
    rec: dict[str, Any] = {
        "address": hex(ins.address), "address_int": ins.address,
        "block": hex(block_start if block_start is not None else ins.address),
        "mnemonic": full_mnemonic, "prefixes": prefixes, "assembly_operands": ins.operands,
        "opcode": "opaque", "destination": None, "defs": [], "inputs": [],
        "expression": None, "width": None, "flow": ins.flow,
        "target": hex(ins.direct_target) if ins.direct_target is not None else None,
        "reference_target": hex(ins.reference_target) if ins.reference_target is not None else None,
        "import_name": ins.import_name, "modeled": False,
        "zeroes_upper_bits": False,
        "input_tokens": tokens,
    }
    def set_op(opcode: str, dest: str | None, inputs: list[dict[str, Any]], expr: str | None = None,
               defs: list[str] | None = None, width: int | None = None) -> dict[str, Any]:
        input_types = [_type_for_token(x, bits) for x in inputs]
        destination_type = _type_for_destination(dest, width, bits)
        rec.update(opcode=opcode, destination=dest, defs=defs if defs is not None else ([dest] if dest else []),
                   inputs=[_value(x) for x in inputs], input_tokens=inputs, expression=expr,
                   input_types=input_types, destination_type=destination_type,
                   type_confidence={
                       "inputs": ["high" if value not in {"unknown", "int"} else "medium" for value in input_types],
                       "destination": "high" if destination_type in {"bool", "flags_t", "uintptr_t"} else "medium" if destination_type else "none",
                   },
                   width=width, modeled=True, atomic=("lock" in prefixes))
        return rec
    def x87_register(index: int) -> dict[str, Any]:
        name = f"ST{index}"
        return {"kind": "var", "value": name, "text": name, "width": 80, "type": "long double"}
    def x87_index(operand: str) -> int | None:
        match = re.fullmatch(r"\s*st(?:\s*\(\s*([0-7])\s*\))?\s*", operand, re.I)
        return int(match.group(1) or 0) if match else None
    fence_order = {
        "lfence": "load_and_execution", "sfence": "store",
        "mfence": "full", "serialize": "serializing",
    }
    if m in fence_order:
        rec = set_op("memory_fence", None, [], f"memory_order_fence({fence_order[m]})",
                     defs=["MEMORY_ORDER"])
        rec["ordering"] = fence_order[m]
        return rec
    if m == "pause":
        return set_op("spin_wait_hint", None, [], "processor_spin_wait_hint", defs=[])
    if m in {"finit", "fninit"}:
        return set_op("x87_init", "X87_STACK", [], "empty_x87_stack", defs=["X87_STACK"], width=80)
    if m in {"fabs", "fchs"}:
        operation = "absolute_value" if m == "fabs" else "negate_sign"
        return set_op("x87_unary", "ST0", [x87_register(0)],
                      f"{operation}(ST0)", defs=["X87_STACK", "ST0"], width=80)
    if m in {"fld1", "fldz", "fldpi"}:
        value = {"fld1": "1.0", "fldz": "0.0", "fldpi": "x87_pi"}[m]
        rec = set_op("x87_push", "ST0", [], value, defs=["X87_STACK", "ST0"], width=80)
        rec["stack_effect"] = 1
        return rec
    if m == "fld" and tokens:
        index = x87_index(operands[0])
        source = x87_register(index) if index is not None else tokens[0]
        rec = set_op("x87_push", "ST0", [source], f"push({_value(source)})", defs=["X87_STACK", "ST0"], width=80)
        rec["stack_effect"] = 1
        return rec
    if m == "fild" and tokens:
        rec = set_op("x87_integer_load", "ST0", [tokens[0]], f"push(signed_int_to_x87({_value(tokens[0])}))",
                     defs=["X87_STACK", "ST0"], width=80)
        rec["stack_effect"] = 1
        return rec
    if m in {"fist", "fistp", "fisttp"} and tokens:
        conversion = "truncate_to_int" if m == "fisttp" else "round_to_int_using_x87_control_word"
        dest = _value(tokens[0])
        rec = set_op("x87_integer_store", dest, [x87_register(0), tokens[0]],
                     f"{dest} = {conversion}(ST0)",
                     defs=[dest] + (["X87_STACK", "ST0"] if m in {"fistp", "fisttp"} else []),
                     width=tokens[0].get("width"))
        rec["stack_effect"] = -1 if m in {"fistp", "fisttp"} else 0
        rec["rounding_mode"] = "truncate" if m == "fisttp" else "current_x87_control_word"
        return rec
    if m in {"fst", "fstp"} and tokens:
        destination_index = x87_index(operands[0])
        destination = f"ST{destination_index}" if destination_index is not None else _value(tokens[0])
        rec = set_op("x87_store", destination, [x87_register(0), tokens[0]],
                     f"{destination} = ST0", defs=[destination] + (["X87_STACK", "ST0"] if m == "fstp" else []),
                     width=tokens[0].get("width"))
        rec["stack_effect"] = -1 if m == "fstp" else 0
        return rec
    if m == "fxch":
        index = x87_index(operands[0]) if operands else 1
        index = 1 if index is None else index
        left, right = x87_register(0), x87_register(index)
        return set_op("x87_exchange", "X87_STACK", [left, right], f"swap(ST0, ST{index})",
                      defs=["X87_STACK", "ST0", f"ST{index}"], width=80)
    if re.fullmatch(r"fcmov(?:b|e|be|u|nb|ne|nbe|nu)", m) and len(tokens) >= 2:
        index = x87_index(operands[1])
        source = x87_register(1 if index is None else index)
        flags = {"kind": "var", "value": "FLAGS", "text": "FLAGS", "width": None}
        return set_op("x87_conditional_copy", "ST0", [x87_register(0), source, flags],
                      f"select(condition:{m[5:]}, {_value(source)}, ST0)",
                      defs=["X87_STACK", "ST0"], width=80)
    if m in {"fadd", "faddp", "fsub", "fsubp", "fsubr", "fsubrp", "fmul", "fmulp", "fdiv", "fdivp", "fdivr", "fdivrp"}:
        pop_after = m.endswith("p")
        base = m[:-1] if pop_after else m
        if len(tokens) >= 2:
            destination_index = x87_index(operands[0])
            source_index = x87_index(operands[1])
            destination = x87_register(destination_index) if destination_index is not None else tokens[0]
            source = x87_register(source_index) if source_index is not None else tokens[1]
        else:
            destination = x87_register(1 if pop_after else 0)
            source = x87_register(0) if pop_after or not tokens else tokens[0]
        operation = {"fadd": "+", "fmul": "*", "fsub": "-", "fsubr": "-", "fdiv": "/", "fdivr": "/"}[base]
        lhs, rhs = (source, destination) if base in {"fsubr", "fdivr"} else (destination, source)
        dest = _value(destination)
        rec = set_op("x87_binary", dest, [lhs, rhs], f"({_value(lhs)} {operation} {_value(rhs)})",
                     defs=[dest] + (["X87_STACK", "ST0"] if pop_after else []), width=80)
        rec["stack_effect"] = -1 if pop_after else 0
        rec["rounding_mode"] = "current_x87_control_word"
        return rec
    if m in {"fcom", "fucom", "fcomp", "fucomp", "fcompp", "fucompp", "fcomi", "fucomi", "fcomip", "fucomip"}:
        pop_count = 2 if m.endswith("pp") else 1 if m.endswith("p") else 0
        compare_index = x87_index(operands[0]) if operands else None
        compare_value = x87_register(compare_index) if compare_index is not None else tokens[0] if tokens else x87_register(1)
        inputs = [x87_register(0), compare_value]
        defs = ["X87_STATUS"] + (["X87_STACK", "ST0"] if pop_count else [])
        if m in {"fcomi", "fucomi", "fcomip", "fucomip"}:
            defs.append("FLAGS")
        rec = set_op("x87_compare", "X87_STATUS", inputs, f"compare({', '.join(_value(x) for x in inputs)})", defs=defs, width=80)
        rec["stack_effect"] = -pop_count
        rec["updates_eflags"] = m in {"fcomi", "fucomi", "fcomip", "fucomip"}
        return rec
    if m in {"nop", "endbr64", "endbr32", "int3"}:
        return set_op("nop" if m != "int3" else "trap", None, [])
    if m in {"cld", "std"}:
        return set_op("direction_flag", "DF", [], f"DF = {1 if m == 'std' else 0}", defs=["DF"], width=1)
    if m in {".byte", ".word", ".long", ".quad"}:
        return rec
    if m in {"ret", "retq", "retn", "retf", "iret", "iretq", "iretd"}:
        return set_op("return", None, [], "return", width=bits)
    if m in {"jmp", "jmpq", "ljmp"}:
        # Direct target is control-flow metadata, not a runtime value use.
        indirect_inputs = tokens if ins.direct_target is None else []
        return set_op("branch", None, indirect_inputs, "goto target", width=None)
    if m in _COND or (m.startswith("j") and m not in {"jmp", "jmpq"}):
        flags = {"kind": "var", "value": "FLAGS", "text": "FLAGS", "width": None}
        count_register, count_width = {
            "jcxz": ("CX", 16), "jecxz": ("ECX", 32), "jrcxz": ("RCX", 64),
        }.get(m, ("RCX", 64) if bits == 64 else ("ECX", 32))
        counter = {"kind": "var", "value": count_register, "text": count_register, "width": count_width}
        if m in {"jecxz", "jrcxz", "jcxz"}:
            return set_op("conditional_branch", None, [counter], f"condition({m})", width=None)
        if m in {"loop", "loope", "loopz", "loopne", "loopnz"}:
            branch_inputs = [counter] + ([flags] if m in {"loope", "loopz", "loopne", "loopnz"} else [])
            rec = set_op("loop_branch", None, branch_inputs, f"decrement_then_condition({m})", defs=[count_register], width=count_width)
            rec["changes_counter"] = True
            return rec
        return set_op("conditional_branch", None, [flags], f"condition({m})", width=None)
    if m in {"call", "callq", "lcall"}:
        # Windows x64 register arguments are visible at the call site. The call writes RAX
        # and clobbers the principal volatile integer registers.
        if bits == 64:
            arg_regs = ["RCX", "RDX", "R8", "R9"]
            clobbers = ["RAX", "RCX", "RDX", "R8", "R9", "R10", "R11"]
        else:
            arg_regs = ["RCX", "RDX"]
            clobbers = ["RAX", "RCX", "RDX"]
        ins_tokens = [{"kind": "var", "value": x, "text": x, "width": bits} for x in arg_regs]
        target_tokens = [] if ins.direct_target is not None or ins.import_name else tokens[:1]
        return set_op("call", None, target_tokens + ins_tokens, "call", defs=clobbers, width=bits)
    if m in {"push", "pop"} and tokens:
        if m == "push":
            rsp = {"kind": "var", "value": "RSP", "text": "RSP", "width": bits}
            return set_op("stack_push", "RSP", tokens + [rsp], "RSP = RSP - stack_width; push", defs=["RSP"], width=bits)
        dst = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        rsp = {"kind": "var", "value": "RSP", "text": "RSP", "width": bits}
        defs = ["RSP"] + ([dst] if dst else [])
        return set_op("stack_pop", dst, [rsp], "pop; RSP = RSP + stack_width", width=tokens[0].get("width"), defs=defs)
    if m == "leave":
        rbp = {"kind": "var", "value": "RBP", "text": "RBP", "width": bits}
        rsp = {"kind": "var", "value": "RSP", "text": "RSP", "width": bits}
        return set_op("leave", "RBP", [rbp, rsp], "RSP = RBP; RBP = pop()", defs=["RSP", "RBP"], width=bits)
    if m in {"cmp", "test", "ucomiss", "comiss", "comisd", "ucomisd"}:
        op = "compare" if m == "cmp" else "bit_test" if m == "test" else "float_compare"
        return set_op(op, None, tokens, m, defs=["FLAGS"], width=tokens[0].get("width") if tokens else None)
    if m in _STRING_INSTRUCTIONS and (m not in {"movsd", "cmpsd"} or not tokens):
        kind, width = _STRING_INSTRUCTIONS[m]
        repeated = any(prefix in {"rep", "repe", "repz", "repne", "repnz"} for prefix in prefixes)
        repeat_condition = next((prefix for prefix in prefixes if prefix in {"repe", "repz", "repne", "repnz"}), None)
        rsi = {"kind": "var", "value": "RSI", "text": "RSI", "width": bits}
        rdi = {"kind": "var", "value": "RDI", "text": "RDI", "width": bits}
        rcx = {"kind": "var", "value": "RCX", "text": "RCX", "width": bits}
        df = {"kind": "var", "value": "DF", "text": "DF", "width": 1}
        source = {"kind": "memory", "value": "MEM[RSI]", "text": "[RSI]", "width": width}
        target = {"kind": "memory", "value": "MEM[RDI]", "text": "[RDI]", "width": width}
        accumulator = {"kind": "var", "value": "RAX", "text": "RAX", "width": width}
        inputs: list[dict[str, Any]] = []
        defs: list[str] = []
        destination: str | None = None
        if kind == "move":
            inputs.extend((source, rsi, rdi, df))
            defs.extend(("MEM[RDI]", "RSI", "RDI"))
            destination = "MEM[RDI]"
        elif kind == "store":
            inputs.extend((accumulator, rdi, df))
            defs.extend(("MEM[RDI]", "RDI"))
            destination = "MEM[RDI]"
        elif kind == "load":
            inputs.extend((source, rsi, df))
            defs.extend(("RAX", "RSI"))
            destination = "RAX"
        elif kind == "compare":
            inputs.extend((source, target, rsi, rdi, df))
            defs.extend(("RSI", "RDI", "FLAGS"))
        elif kind == "scan":
            inputs.extend((accumulator, target, rdi, df))
            defs.extend(("RDI", "FLAGS"))
        elif kind == "input":
            inputs.extend((rdi, df))
            defs.extend(("MEM[RDI]", "RDI"))
            destination = "MEM[RDI]"
        else:  # output
            inputs.extend((source, rsi, df))
            defs.append("RSI")
        if repeated:
            inputs.append(rcx)
            defs.append("RCX")
            if repeat_condition:
                inputs.append({"kind": "var", "value": "FLAGS", "text": "FLAGS", "width": None})
        operation = "string_" + kind
        rec = set_op(operation, destination, inputs,
                     f"{full_mnemonic}: {kind} {width}-bit element with direction from DF",
                     defs=defs, width=width)
        rec["repeat_prefix"] = next((p for p in prefixes if p in {"rep", "repe", "repz", "repne", "repnz"}), None)
        rec["repeat_condition"] = repeat_condition
        rec["atomic"] = "lock" in prefixes
        return rec
    if m in {"bt", "bts", "btr", "btc"} and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        defs = ["FLAGS"]
        if m != "bt" and destination:
            defs.insert(0, destination)
        return set_op("bit_test", destination, tokens, f"{m}({_value(tokens[0])}, {_value(tokens[1])})",
                      defs=defs, width=tokens[0].get("width"))
    if m in {"rol", "ror", "rcl", "rcr"} and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] in {"var", "memory"} else None
        inputs = tokens[:2] + ([{"kind": "var", "value": "FLAGS", "text": "FLAGS", "width": None}]
                               if m in {"rcl", "rcr"} else [])
        return set_op("rotate", destination, inputs, f"{m}({_value(tokens[0])}, {_value(tokens[1])})",
                      defs=([destination] if destination else []) + ["FLAGS"], width=tokens[0].get("width"))
    if m in {"cdq", "cqo", "cwd"}:
        source = {"kind": "var", "value": "RAX", "text": "RAX", "width": bits}
        destination = "RDX"
        return set_op("sign_extend", destination, [source], f"sign_extend({source['value']})",
                      defs=["RDX"], width=bits)
    if m == "lea" and len(tokens) >= 2:
        dst = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        src = tokens[1]
        src_addr = {
            "kind": "opaque", "value": f"address_of({src['value']})",
            "text": src["text"], "width": src.get("width"), "type": "uintptr_t",
        }
        if ins.reference_target is not None:
            src_addr = {"kind": "address", "value": hex(ins.reference_target), "text": src["text"], "width": bits}
        return set_op("address", dst, [src_addr], "address_of", width=tokens[0].get("width"))
    if m == "xchg" and len(tokens) >= 2:
        if any(token["kind"] == "memory" for token in tokens[:2]):
            destination = _value(tokens[0])
            rec = set_op("atomic_exchange", destination, tokens[:2],
                         f"atomic_exchange({destination}, {_value(tokens[1])})",
                         defs=[_value(tokens[0]), _value(tokens[1])], width=tokens[0].get("width"))
            rec["atomic"] = True  # Memory XCHG is implicitly locked.
            return rec
        if tokens[0]["kind"] != "var" or tokens[1]["kind"] != "var":
            return set_op("exchange", None, tokens, "exchange", width=tokens[0].get("width"))
        left, right = _value(tokens[0]), _value(tokens[1])
        return set_op("exchange", left, tokens, f"swap({left}, {right})",
                      defs=[left, right], width=tokens[0].get("width"))
    if m == "xadd" and len(tokens) >= 2:
        destination, source = _value(tokens[0]), _value(tokens[1])
        return set_op("exchange_add", destination, tokens[:2],
                      f"old_{destination} = {source}; {destination} = {destination} + {source}",
                      defs=[destination, source, "FLAGS"], width=tokens[0].get("width"))
    if m == "cmpxchg" and len(tokens) >= 2:
        accumulator = {"kind": "var", "value": "RAX", "text": "RAX", "width": tokens[0].get("width")}
        destination, source = _value(tokens[0]), _value(tokens[1])
        opcode = "atomic_compare_exchange" if tokens[0]["kind"] == "memory" else "compare_exchange"
        return set_op(opcode, destination, [accumulator, tokens[0], tokens[1]],
                      f"if RAX == {destination}: {destination} = {source}; else: RAX = {destination}",
                      defs=[destination, "RAX", "FLAGS"], width=tokens[0].get("width"))
    if m in {"mul", "imul"} and len(tokens) == 1:
        rax = {"kind": "var", "value": "RAX", "text": "RAX", "width": bits}
        rdx = {"kind": "var", "value": "RDX", "text": "RDX", "width": bits}
        return set_op("wide_multiply", "RAX", [rax, tokens[0]], "RDX:RAX = RAX * operand",
                      defs=["RAX", "RDX", "FLAGS"], width=tokens[0].get("width"))
    if m == "bswap" and tokens and tokens[0]["kind"] == "var" and tokens[0].get("width") in {32, 64}:
        destination = _value(tokens[0])
        return set_op("byte_swap", destination, tokens[:1], f"bswap({destination})",
                      defs=[destination], width=tokens[0].get("width"))
    if m in {"popcnt", "lzcnt", "tzcnt", "bsf", "bsr"} and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        opcode, expression = {
            "popcnt": ("population_count", "popcount"),
            "lzcnt": ("leading_zero_count", "leading_zeros"),
            "tzcnt": ("trailing_zero_count", "trailing_zeros"),
            "bsf": ("bit_scan_forward", "bit_scan_forward"),
            "bsr": ("bit_scan_reverse", "bit_scan_reverse"),
        }[m]
        rec = set_op(opcode, destination, tokens[1:],
                     f"{expression}({_value(tokens[1])})",
                     defs=([destination, "FLAGS"] if destination else ["FLAGS"]),
                     width=tokens[0].get("width"))
        rec["zero_input_result"] = "operand_width" if m in {"lzcnt", "tzcnt"} else "undefined" if m in {"bsf", "bsr"} else 0
        rec["zero_flag"] = "source_is_zero" if m in {"popcnt", "bsf", "bsr"} else "result_is_zero"
        if m in {"lzcnt", "tzcnt"}:
            rec["carry_flag"] = "source_is_zero"
        return rec
    if m == "andn" and len(tokens) >= 3:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        rec = set_op("bitwise_and_not", destination, tokens[1:3],
                     f"(~{_value(tokens[1])} & {_value(tokens[2])})",
                     defs=([destination, "FLAGS"] if destination else ["FLAGS"]),
                     width=tokens[0].get("width"))
        rec["carry_flag"] = 0
        rec["overflow_flag"] = 0
        return rec
    if m in {"blsi", "blsmsk", "blsr"} and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        operation = {"blsi": "isolate_lowest_set_bit", "blsmsk": "mask_through_lowest_set_bit",
                     "blsr": "reset_lowest_set_bit"}[m]
        rec = set_op("bit_manipulation", destination, tokens[1:2],
                     f"{operation}({_value(tokens[1])})",
                     defs=([destination, "FLAGS"] if destination else ["FLAGS"]),
                     width=tokens[0].get("width"))
        rec["carry_flag"] = "source_is_zero"
        rec["overflow_flag"] = 0
        return rec
    if m in {"div", "idiv"} and tokens:
        rax = {"kind": "var", "value": "RAX", "text": "RAX", "width": bits}
        rdx = {"kind": "var", "value": "RDX", "text": "RDX", "width": bits}
        return set_op("divide", "RAX", [rdx, rax, tokens[0]], "RAX = quotient; RDX = remainder",
                      defs=["RAX", "RDX"], width=tokens[0].get("width"))
    if (re.fullmatch(r"v?pextr[bwdq]", m) or m in {"extractps", "vextractps"}) and len(tokens) >= 3:
        source = _value(tokens[1])
        destination = _value(tokens[0])
        rec = set_op("vector_extract", destination, tokens[1:],
                     f"extract_lane({source}, {_value(tokens[2])})",
                     defs=[destination], width=tokens[0].get("width"))
        rec["lane_width"] = {"b": 8, "w": 16, "d": 32, "q": 64}.get(m[-1], 32)
        return rec
    if re.fullmatch(r"v?pinsr[bwdq]", m) and len(tokens) >= 3:
        destination = _value(tokens[0])
        vector = tokens[1] if m.startswith("v") else tokens[0]
        scalar = tokens[2] if m.startswith("v") else tokens[1]
        rec = set_op("vector_insert", destination,
                     [vector, scalar, tokens[-1]],
                     f"insert_lane({_value(vector)}, {_value(scalar)}, {_value(tokens[-1])})",
                     defs=[destination], width=tokens[0].get("width"))
        rec["lane_width"] = {"b": 8, "w": 16, "d": 32, "q": 64}[m[-1]]
        return rec
    if re.fullmatch(r"v?cvtsi2(?:ss|sd)", m) and len(tokens) >= (3 if m.startswith("v") else 2):
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") else tokens[1:2]
        precision = "double" if m.endswith("sd") else "single"
        rec = set_op("scalar_int_to_float", destination, inputs,
                     f"signed_int_to_{precision}({_value(tokens[-1])})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["precision"] = precision
        rec["lane_width"] = 64 if precision == "double" else 32
        rec["constant_fold_requires_exact_result"] = True
        return rec
    if re.fullmatch(r"v?cvtt?(?:ss|sd)2si", m) and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        precision = "double" if "sd2si" in m else "single"
        truncates = "tt" in m
        rec = set_op("scalar_float_to_int", destination, tokens[1:],
                     f"float_to_signed_int({_value(tokens[1])})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["precision"] = precision
        rec["lane_width"] = 64 if precision == "double" else 32
        rec["rounding"] = "truncate_toward_zero" if truncates else "fold_only_integral_values"
        return rec
    if re.fullmatch(r"v?cvt(?:ss2sd|sd2ss)", m) and len(tokens) >= (3 if m.startswith("v") else 2):
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        source = tokens[-1]
        source_precision = "single" if m.endswith("ss2sd") else "double"
        target_precision = "double" if source_precision == "single" else "single"
        inputs = tokens[1:] if m.startswith("v") else [source]
        rec = set_op("scalar_float_convert", destination, inputs,
                     f"{source_precision}_to_{target_precision}({_value(source)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["source_precision"] = source_precision
        rec["target_precision"] = target_precision
        rec["constant_fold_requires_exact_result"] = True
        rec["preserves_upper_scalar_lanes"] = True
        rec["mxcsr_rounding_needed_for_inexact_narrowing"] = source_precision == "double"
        return rec
    if re.fullmatch(r"v?round(?:ss|sd)", m) and len(tokens) >= (4 if m.startswith("v") else 3):
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        precision = "double" if m.endswith("sd") else "single"
        rec = set_op("scalar_float_round", destination, tokens[1:],
                     f"round_{precision}({_value(tokens[-2])}, {_value(tokens[-1])})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["precision"] = precision
        rec["lane_width"] = 64 if precision == "double" else 32
        rec["rounding_control"] = "immediate[1:0]"
        rec["mxcsr_rounding_when_immediate_bit_2_is_clear"] = True
        rec["preserves_upper_scalar_lanes"] = True
        rec["constant_fold_requires_explicit_rounding_mode"] = True
        return rec
    if re.fullmatch(r"v?(?:min|max)(?:ss|sd)", m) and len(tokens) >= (3 if m.startswith("v") else 2):
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") else tokens[:2]
        operation = m[1:4] if m.startswith("v") else m[:3]
        precision = "double" if m.endswith("sd") else "single"
        rec = set_op("scalar_float_minmax", destination, inputs,
                     f"{operation}_{precision}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["precision"] = precision
        rec["lane_width"] = 64 if precision == "double" else 32
        rec["unordered_or_equal_selects_second_source"] = True
        rec["preserves_upper_scalar_lanes"] = True
        return rec
    if re.fullmatch(r"v?(?:add|sub|mul|div)(?:ss|sd)", m) and len(tokens) >= (3 if m.startswith("v") else 2):
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") else tokens[:2]
        operation = m[1:4] if m.startswith("v") else m[:3]
        precision = "double" if m.endswith("sd") else "single"
        rec = set_op("scalar_float_binary", destination, inputs,
                     f"{operation}_{precision}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["precision"] = precision
        rec["lane_width"] = 64 if precision == "double" else 32
        rec["preserves_upper_scalar_lanes"] = True
        rec["constant_fold_requires_exact_result"] = True
        return rec
    if re.fullmatch(r"v?cvt(?:ps2pd|pd2ps)", m) and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        widening = m.endswith("ps2pd")
        source_precision, target_precision = ("single", "double") if widening else ("double", "single")
        rec = set_op("vector_float_convert", destination, tokens[1:],
                     f"{source_precision}_to_{target_precision}({_value(tokens[1])})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["source_precision"] = source_precision
        rec["target_precision"] = target_precision
        rec["source_lane_width"] = 32 if widening else 64
        rec["destination_lane_width"] = 64 if widening else 32
        rec["constant_fold_requires_exact_result"] = True
        rec["nan_payload_conversion_modeled"] = False
        return rec
    if re.fullmatch(r"v?(?:cvtdq2ps|cvttps2dq|cvtps2dq)", m) and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        operation = re.fullmatch(r"v?(cvtdq2ps|cvttps2dq|cvtps2dq)", m).group(1)
        if operation == "cvtdq2ps":
            source_type, target_type = "signed_int32", "float32"
            rounding = "mxcsr; fold exact results only"
        else:
            source_type, target_type = "float32", "signed_int32"
            rounding = "truncate_toward_zero" if operation == "cvttps2dq" else "mxcsr; fold integral values only"
        rec = set_op("vector_float_int_convert", destination, tokens[1:],
                     f"{operation}({_value(tokens[1])})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["source_element_type"] = source_type
        rec["target_element_type"] = target_type
        rec["lane_width"] = 32
        rec["rounding"] = rounding
        rec["out_of_range_behavior"] = "unknown"
        return rec
    if re.fullmatch(r"v?(?:min|max)(?:ps|pd)", m) and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") and len(tokens) >= 3 else tokens[:2]
        operation = re.fullmatch(r"v?(min|max)(ps|pd)", m).groups()
        precision = "double" if operation[1] == "pd" else "single"
        rec = set_op("vector_float_minmax", destination, inputs,
                     f"{operation[0]}_{precision}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["precision"] = precision
        rec["lane_width"] = 64 if precision == "double" else 32
        rec["unordered_or_equal_selects_second_source"] = True
        rec["preserves_operand_bit_patterns"] = True
        rec["exception_flags_modeled"] = False
        return rec
    if re.fullmatch(r"v?sqrt(?:ps|pd)", m) and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        precision = "double" if m.endswith("pd") else "single"
        rec = set_op("vector_float_sqrt", destination, tokens[1:2],
                     f"sqrt_{precision}({_value(tokens[1])})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["precision"] = precision
        rec["lane_width"] = 64 if precision == "double" else 32
        rec["constant_fold_requires_exact_nonnegative_result"] = True
        rec["exception_flags_modeled"] = False
        return rec
    if re.fullmatch(r"v?(?:add|sub|mul|div)(?:ps|pd)", m) and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") and len(tokens) >= 3 else tokens[:2]
        operation = re.fullmatch(r"v?(add|sub|mul|div)(ps|pd)", m).groups()
        precision = "double" if operation[1] == "pd" else "single"
        rec = set_op("vector_float_binary", destination, inputs,
                     f"{operation[0]}_{precision}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["precision"] = precision
        rec["lane_width"] = 64 if precision == "double" else 32
        rec["lane_wise"] = True
        rec["constant_fold_requires_exact_finite_result"] = True
        rec["exception_flags_modeled"] = False
        return rec
    if re.fullmatch(r"v?cmpp[sd]", m) and len(tokens) >= 3:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:]
        lane_width = 64 if m.endswith("pd") else 32
        rec = set_op("vector_float_compare", destination, inputs,
                     f"{m}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["precision"] = "double" if lane_width == 64 else "single"
        rec["lane_width"] = lane_width
        rec["predicate_operand"] = _value(tokens[-1])
        rec["supported_predicate_range"] = [0, 31 if m.startswith("v") else 7]
        rec["comparison_result"] = "all_ones_or_zero_per_lane"
        rec["nan_ordering"] = "explicit_ordered_unordered_predicates"
        return rec
    if re.fullmatch(r"v?(?:movmskps|movmskpd|pmovmskb)", m) and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        source = tokens[1]
        element_width = 32 if "ps" in m else 64 if "pd" in m else 8
        rec = set_op("vector_movemask", destination, [source],
                     f"extract_sign_bits({_value(source)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["element_width"] = element_width
        rec["result_bit_count"] = (source.get("width") or 0) // element_width
        return rec
    if re.fullmatch(r"v?ptest", m) and len(tokens) >= 2:
        inputs = tokens[1:] if m.startswith("v") and len(tokens) >= 3 else tokens[:2]
        rec = set_op("vector_test_flags", None, inputs,
                     f"test_vector_bits({', '.join(_value(x) for x in inputs)})",
                     defs=["FLAGS"], width=tokens[0].get("width"))
        rec["flags_defined"] = ["CF", "ZF"]
        rec["carry_flag"] = "no_bits_in_second_outside_first"
        rec["zero_flag"] = "no_common_set_bits"
        rec["other_arithmetic_flags"] = "undefined"
        return rec
    if re.fullmatch(r"v?palignr", m) and len(tokens) >= 3:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") and len(tokens) >= 4 else tokens[:3]
        rec = set_op("vector_align", destination, inputs,
                     f"{m}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["lane_width"] = 128
        rec["control"] = "immediate_byte_count"
        rec["operates_per_128_bit_lane"] = True
        rec["zeroes_lane_when_shift_at_least"] = 32
        return rec
    if re.fullmatch(r"v?pblendvb", m) and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        implicit_mask = not (m.startswith("v") and len(tokens) >= 4)
        inputs = tokens[1:] if not implicit_mask else tokens[:2] + [normalize_operand("xmm0", ins)]
        rec = set_op("vector_blend", destination, inputs,
                     f"{m}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["lane_width"] = 8
        rec["control"] = "mask_byte_sign_bit"
        rec["implicit_mask_register"] = "XMM0" if implicit_mask else None
        return rec
    if re.fullmatch(r"v?(?:blendps|blendpd|pblendw|pblendd)", m) and len(tokens) >= 3:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") and len(tokens) >= 4 else tokens[:3]
        operation = re.fullmatch(r"v?(blendps|blendpd|pblendw|pblendd)", m).group(1)
        lane_width = {"blendps": 32, "blendpd": 64, "pblendw": 16, "pblendd": 32}[operation]
        rec = set_op("vector_blend", destination, inputs,
                     f"{m}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["lane_width"] = lane_width
        rec["control"] = "immediate_bit_per_lane"
        rec["selector_bit_count"] = 8
        rec["selector_repeats_per_128_bit_lane"] = operation == "pblendw"
        return rec
    if re.fullmatch(r"v?pshufb|v?pshuf[hl]w|v?pshufd|v?shufps|v?shufpd", m) and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        if m.endswith("pshufb"):
            inputs = tokens[1:] if m.startswith("v") and len(tokens) >= 3 else tokens[:2]
            lane_width, shuffle_kind = 8, "byte_control"
        elif m.endswith(("pshufd", "pshuflw", "pshufhw")):
            inputs = [tokens[1], tokens[2]] if len(tokens) >= 3 else tokens[1:]
            lane_width = 16 if m.endswith(("pshuflw", "pshufhw")) else 32
            shuffle_kind = "immediate"
        else:
            inputs = tokens[1:] if m.startswith("v") else tokens[:3]
            lane_width = 64 if m.endswith("shufpd") else 32
            shuffle_kind = "immediate"
        rec = set_op("vector_shuffle", destination, inputs,
                     f"{m}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["lane_width"] = lane_width
        rec["shuffle_control"] = shuffle_kind
        rec["operates_per_128_bit_lane"] = True
        if m.endswith("pshufb"):
            rec["control_high_bit_zeroes_output"] = True
        return rec
    if re.fullmatch(r"v?pack(?:sswb|ssdw|uswb|usdw)", m) and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") and len(tokens) >= 3 else tokens[:2]
        operation = m[1:] if m.startswith("v") else m
        rec = set_op("vector_pack_saturate", destination, inputs,
                     f"{m}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["input_lane_width"] = 32 if operation.endswith("sdw") else 16
        rec["output_lane_width"] = rec["input_lane_width"] // 2
        rec["signed_output"] = operation.startswith("packss")
        rec["operates_per_128_bit_lane"] = True
        return rec
    if re.fullmatch(r"v?(?:pmaddwd|pmaddubsw|psadbw)", m) and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") and len(tokens) >= 3 else tokens[:2]
        operation = m[1:] if m.startswith("v") else m
        output_width, input_width, behavior = {
            "pmaddwd": (32, 16, "signed_pairwise_products_sum_wrap"),
            "pmaddubsw": (16, 8, "unsigned_signed_pairwise_products_sum_saturate_signed"),
            "psadbw": (64, 8, "sum_absolute_unsigned_byte_differences"),
        }[operation]
        rec = set_op("vector_pairwise_arithmetic", destination, inputs,
                     f"{m}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["input_lane_width"] = input_width
        rec["output_lane_width"] = output_width
        rec["pairwise_behavior"] = behavior
        rec["operates_per_128_bit_lane"] = operation == "psadbw"
        return rec
    horizontal = re.fullmatch(r"v?ph(add|sub)(sw|w|d)", m)
    if horizontal and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") and len(tokens) >= 3 else tokens[:2]
        operation, suffix = horizontal.groups()
        lane_width = 32 if suffix == "d" else 16
        rec = set_op("vector_horizontal_arithmetic", destination, inputs,
                     f"{m}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["lane_width"] = lane_width
        rec["operation"] = operation
        rec["signed_saturating"] = suffix == "sw"
        rec["operates_per_128_bit_lane"] = True
        return rec
    sign_operation = re.fullmatch(r"v?psign([bwd])", m)
    if sign_operation and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") and len(tokens) >= 3 else tokens[:2]
        rec = set_op("vector_sign", destination, inputs,
                     f"{m}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["lane_width"] = {"b": 8, "w": 16, "d": 32}[sign_operation.group(1)]
        rec["signed_control_lane"] = True
        rec["zero_control_zeroes_lane"] = True
        return rec
    absolute = re.fullmatch(r"v?pabs([bwd])", m)
    if absolute and tokens:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        source = tokens[1:2]
        rec = set_op("vector_absolute", destination, source,
                     f"{m}({_value(source[0])})" if source else m,
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["lane_width"] = {"b": 8, "w": 16, "d": 32}[absolute.group(1)]
        rec["signed_input"] = True
        rec["minimum_signed_lane_wraps"] = True
        return rec
    if re.fullmatch(r"v?punpck[lh](?:bw|wd|dq|qdq)", m) and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") and len(tokens) >= 3 else tokens[:2]
        suffix = re.fullmatch(r"v?punpck[lh](bw|wd|dq|qdq)", m).group(1)
        rec = set_op("vector_unpack", destination, inputs,
                     f"{m}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["lane_width"] = {"bw": 8, "wd": 16, "dq": 32, "qdq": 64}[suffix]
        rec["source_half"] = "high" if "punpckh" in m else "low"
        rec["operates_per_128_bit_lane"] = True
        return rec
    if (re.fullmatch(r"v?pmul(?:lw|hw|huw|ld|dq|udq)", m) and len(tokens) >= 2):
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") and len(tokens) >= 3 else tokens[:2]
        operation = m[1:] if m.startswith("v") else m
        lane_width = 32 if operation.endswith("ld") else 64 if operation.endswith(("dq", "udq")) else 16
        rec = set_op("vector_multiply", destination, inputs,
                     f"{m}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["lane_width"] = lane_width
        rec["result_selection"] = "high_half" if operation.endswith(("hw", "huw")) else "low_half" if operation.endswith("lw") else "full_even_dword_products" if operation.endswith(("dq", "udq")) else "full_lane_product_truncated"
        rec["signed"] = operation in {"pmullw", "pmulhw", "pmulld", "pmuldq"}
        return rec
    if (m in {"pxor", "vpxor", "por", "vpor", "pand", "vpand", "pandn", "vpandn"}
            or re.fullmatch(r"v?p(?:add|sub)[bwdq]", m)
            or re.fullmatch(r"v?ps(?:ll|rl|ra)[wdq]", m)
            or re.fullmatch(r"v?ps(?:ll|rl)dq", m)
            or re.fullmatch(r"v?p(?:add|sub)(?:us|s)[bw]", m)
            or re.fullmatch(r"v?pavg[bw]", m)
            or re.fullmatch(r"v?pcmp(?:eq|gt)[bwdq]", m)
            or re.fullmatch(r"v?p(?:min|max)[us][bwd]", m)) and len(tokens) >= 2:
        destination = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        inputs = tokens[1:] if m.startswith("v") and len(tokens) >= 3 else tokens[:2]
        lane_match = re.fullmatch(r"v?p(?:add|sub)([bwdq])", m)
        shift_match = re.fullmatch(r"v?ps(?:ll|rl|ra)([wdq])", m)
        byte_shift_match = re.fullmatch(r"v?ps(?:ll|rl)dq", m)
        lane_width = {"b": 8, "w": 16, "d": 32, "q": 64}.get(lane_match.group(1)) if lane_match else None
        if shift_match:
            lane_width = {"w": 16, "d": 32, "q": 64}[shift_match.group(1)]
        elif byte_shift_match:
            lane_width = 128
        saturating_match = re.fullmatch(r"v?p(?:add|sub)(?:us|s)([bw])", m)
        average_match = re.fullmatch(r"v?pavg([bw])", m)
        compare_match = re.fullmatch(r"v?pcmp(?:eq|gt)([bwdq])", m)
        extrema_match = re.fullmatch(r"v?p(?:min|max)[us]([bwd])", m)
        if saturating_match:
            lane_width = {"b": 8, "w": 16}[saturating_match.group(1)]
        elif average_match:
            lane_width = {"b": 8, "w": 16}[average_match.group(1)]
        elif compare_match:
            lane_width = {"b": 8, "w": 16, "d": 32, "q": 64}[compare_match.group(1)]
        elif extrema_match:
            lane_width = {"b": 8, "w": 16, "d": 32}[extrema_match.group(1)]
        opcode = "vector_shift" if shift_match or byte_shift_match else "vector_compare" if compare_match else "vector_binary"
        rec = set_op(opcode, destination, inputs, f"{m}({', '.join(_value(x) for x in inputs)})",
                     defs=[destination] if destination else [], width=tokens[0].get("width"))
        rec["lane_width"] = lane_width
        return rec
    if m in {
        "mov", "movabs", "movzx", "movsx", "movsxd", "movs", "movd", "movq",
        "movdqa", "movdqu", "movaps", "movups", "movapd", "movupd",
        "vmovd", "vmovq", "vmovdqa", "vmovdqu", "vmovaps", "vmovups", "vmovapd", "vmovupd",
    } and len(tokens) >= 2:
        dst = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        if dst is None:
            return set_op("store", None, tokens, "store", width=tokens[0].get("width"))
        src_kind = "zero_extend" if (m == "movzx" or m in {"movd", "vmovd", "movq", "vmovq"}
                                     and (tokens[0].get("width") or 0) > (tokens[1].get("width") or 0)
                                     or tokens[0].get("width") == 64 and tokens[1].get("width") == 32
                                     and tokens[0]["kind"] == "var") else "sign_extend" if m in {"movsx", "movsxd"} else "copy"
        expr = f"{src_kind}({_value(tokens[1])})" if src_kind.endswith("extend") else _value(tokens[1])
        rec = set_op(src_kind, dst, [tokens[1]], expr, width=tokens[0].get("width"))
        rec["zeroes_upper_bits"] = bool(bits == 64 and tokens[0].get("width") == 32 and tokens[0]["kind"] == "var" and raw_destination_is_32bit(operands[0]))
        return rec
    if m in {"shld", "shrd"} and len(tokens) >= 3:
        destination = _value(tokens[0])
        rec = set_op(
            "double_shift", destination, tokens[:3],
            f"{m}({_value(tokens[0])}, {_value(tokens[1])}, {_value(tokens[2])})",
            defs=[destination, "FLAGS"], width=tokens[0].get("width"),
        )
        rec["count_mask"] = 0x3F if bits == 64 and tokens[0].get("width") == 64 else 0x1F
        rec["preserves_flags_when_count_zero"] = True
        rec["result_undefined_when_count_exceeds_width"] = True
        return rec
    if m in _BINARY and len(tokens) >= 2:
        if tokens[0]["kind"] != "var":
            return set_op("store_compute", None, tokens, f"memory {m}", width=tokens[0].get("width"))
        dst = _value(tokens[0])
        if m == "imul" and len(tokens) >= 3:
            lhs, rhs = tokens[1], tokens[2]
        else:
            lhs, rhs = tokens[0], tokens[1]
        op = _BINARY[m]
        if m == "xor" and len(tokens) >= 2 and _value(tokens[0]) == _value(tokens[1]):
            expr = "const:0"
        else:
            expr = f"({_value(lhs)} {op} {_value(rhs)})"
        inputs = [lhs, rhs]
        if m in {"adc", "sbb"}:
            inputs.append({"kind": "var", "value": "FLAGS", "text": "FLAGS", "width": None})
        return set_op("binary", dst, inputs, expr, defs=[dst, "FLAGS"], width=tokens[0].get("width"))
    if m in _UNARY and tokens:
        dst = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        defs = [dst, "FLAGS"] if dst and m != "not" else ([dst] if dst else [])
        return set_op("unary", dst, tokens[:1], f"{_UNARY[m]}({_value(tokens[0])})", defs=defs, width=tokens[0].get("width"))
    if m.startswith("set") and tokens:
        dst = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        flags = {"kind": "var", "value": "FLAGS", "text": "FLAGS", "width": None}
        rec = set_op("set_condition", dst, [flags], f"bool(condition:{m[3:]})", width=8)
        rec["destination_type"] = "bool"
        return rec
    if m.startswith("cmov") and len(tokens) >= 2:
        dst = _value(tokens[0]) if tokens[0]["kind"] == "var" else None
        flags = {"kind": "var", "value": "FLAGS", "text": "FLAGS", "width": None}
        return set_op("conditional_copy", dst, [tokens[0], tokens[1], flags], f"select(condition:{m[4:]}, {_value(tokens[1])}, {_value(tokens[0])})", width=tokens[0].get("width"))
    # Keep every unhandled instruction, its operands and flow metadata so the analysis is loss-aware.
    rec["inputs"] = [_value(x) for x in tokens]
    rec["input_tokens"] = tokens
    rec["width"] = tokens[0].get("width") if tokens else None
    rec["opcode"] = "opaque"
    # Preserve a conservative write hint for SSA and downstream consumers. For
    # unknown instructions the first operand is commonly a read/write destination
    # (including SIMD and atomic forms); memory operands may also write memory.
    first = tokens[0] if tokens else None
    destination = None
    if first and first.get("kind") == "var":
        destination = str(first["value"])
    elif first and first.get("kind") == "memory":
        destination = str(first["value"])
    rec["destination"] = destination
    rec["defs"] = [destination] if destination else []
    rec["unknown_effects"] = True
    rec["may_write_memory"] = any(token.get("kind") == "memory" for token in tokens)
    rec["may_write_flags"] = True
    return rec


_CONDITION_FLAGS = {
    "je": {"ZF": True}, "jz": {"ZF": True},
    "jne": {"ZF": False}, "jnz": {"ZF": False},
    "ja": {"CF": False, "ZF": False}, "jnbe": {"CF": False, "ZF": False},
    "jae": {"CF": False}, "jnb": {"CF": False}, "jnc": {"CF": False},
    "jb": {"CF": True}, "jc": {"CF": True}, "jnae": {"CF": True},
    "jbe": {"CF": True, "ZF": True}, "jna": {"CF": True, "ZF": True},
    "jg": {"ZF": False, "SF_equals_OF": True}, "jnle": {"ZF": False, "SF_equals_OF": True},
    "jge": {"SF_equals_OF": True}, "jnl": {"SF_equals_OF": True},
    "jl": {"SF_equals_OF": False}, "jnge": {"SF_equals_OF": False},
    "jle": {"ZF": True, "SF_equals_OF": False}, "jng": {"ZF": True, "SF_equals_OF": False},
    "jo": {"OF": True}, "jno": {"OF": False}, "js": {"SF": True}, "jns": {"SF": False},
    "jp": {"PF": True}, "jpe": {"PF": True}, "jnp": {"PF": False}, "jpo": {"PF": False},
}


def _semantic_register(token: str) -> str | None:
    low = token.lower()
    family = _family(low)
    if family:
        return family
    if low in {"eflags", "rflags", "flags"}:
        return "FLAGS"
    if low in {"df", "x87_stack", "x87_status", "memory_order"}:
        return low.upper()
    return None


def _semantic_effects(ins: Instruction, row: dict[str, Any], bits: int) -> dict[str, Any]:
    """Attach uniform machine-effect fields to the established normalized operation."""
    tokens = row.get("input_tokens") or []
    reads: set[str] = set()
    widths: list[int | None] = []
    memory_reads: list[dict[str, Any]] = []
    for token_index, token in enumerate(tokens):
        widths.append(token.get("width"))
        value = str(token.get("value", ""))
        reg = _semantic_register(value)
        if reg:
            reads.add(reg)
        if token.get("kind") == "memory":
            if not (row.get("opcode") == "store" and token_index == 0):
                memory_reads.append({"address": value, "width_bits": token.get("width"), "certainty": "address_expression"})
            for name in re.findall(r"(?<![A-Za-z0-9_])(?:r(?:1[0-5]|[89])(?:[dwb])?|[er]?(?:ax|bx|cx|dx|si|di|bp|sp|ip)|[abcd][lh])(?![A-Za-z0-9_])", value, re.I):
                family = _semantic_register(name)
                if family:
                    reads.add(family)
    defs = [str(item) for item in row.get("defs", [])]
    writes = {_semantic_register(item) for item in defs}
    writes.discard(None)
    memory_writes = [
        {"address": item, "width_bits": row.get("width"), "certainty": "modeled_destination"}
        for item in defs if item.startswith("MEM[")
    ]
    opcode = str(row.get("opcode", "opaque"))
    if opcode == "store" and tokens and tokens[0].get("kind") == "memory":
        memory_writes.append({"address": str(tokens[0].get("value")),
                              "width_bits": tokens[0].get("width"),
                              "certainty": "modeled_destination"})
    modeled = bool(row.get("modeled")) and opcode != "opaque"
    if opcode in {"call"}:
        memory_reads.append({"address": "unknown_callee_memory", "width_bits": None, "certainty": "conservative_call_effect"})
        memory_writes.append({"address": "unknown_callee_memory", "width_bits": None, "certainty": "conservative_call_effect"})
    if not modeled and "[" in ins.operands:
        memory_reads.append({"address": "unknown_effective_address", "width_bits": None, "certainty": "opaque_instruction"})
        memory_writes.append({"address": "unknown_effective_address", "width_bits": None, "certainty": "opaque_instruction"})
    flags_read = "FLAGS" in reads
    flags_written = "FLAGS" in writes
    if not modeled:
        flags_effect = {"reads": [], "writes": [], "preserves": [], "unknown": True}
    elif flags_written:
        mnemonic = ins.mnemonic.lower().split()[-1]
        arithmetic = {"add", "adc", "sub", "sbb", "cmp", "neg"}
        inc_dec = {"inc", "dec"}
        logic = {"and", "or", "xor", "test", "andn"}
        bit_test = {"bt", "bts", "btr", "btc"}
        if mnemonic in arithmetic:
            defined = ["CF", "PF", "AF", "ZF", "SF", "OF"]
            undefined: list[str] = []
        elif mnemonic in inc_dec:
            defined, undefined = ["PF", "AF", "ZF", "SF", "OF"], []
        elif mnemonic in logic:
            defined, undefined = ["CF", "PF", "ZF", "SF", "OF"], ["AF"]
        elif mnemonic in bit_test:
            defined, undefined = ["CF"], ["PF", "AF", "ZF", "SF", "OF"]
        else:
            # Shifts, rotates, multiply, and architecture-specific flag results
            # vary by count/width; record only that flags change without claiming
            # a complete per-bit result.
            defined, undefined = [], []
        read_flags = ["CF"] if mnemonic in {"adc", "sbb", "rcl", "rcr"} else []
        preserved = ["CF"] if mnemonic in inc_dec else []
        flags_effect = {"reads": read_flags, "writes": defined,
                        "undefined": undefined, "preserves": preserved,
                        "may_write_unspecified": not (defined or undefined), "unknown": False}
    else:
        flags_effect = {"reads": ["CF", "PF", "AF", "ZF", "SF", "OF"] if flags_read else [],
                        "writes": [], "preserves": ["CF", "PF", "AF", "ZF", "SF", "OF"], "unknown": False}
    flow = ins.flow
    control = {
        "kind": flow,
        "direct_target": hex(ins.direct_target) if ins.direct_target is not None else None,
        "indirect": flow in {"call", "jump"} and ins.direct_target is None,
        "condition": _CONDITION_FLAGS.get(ins.mnemonic.lower()) if flow == "conditional" else None,
        "return_stack_bytes": bits // 8 + (int(row.get("input_tokens", [{}])[0].get("value", 0)) if ins.mnemonic.lower() in {"ret", "retf"} and row.get("input_tokens") and row["input_tokens"][0].get("kind") == "const" else 0) if flow == "return" else 0,
    }
    return {
        "opcode": opcode,
        "operation_type": opcode,
        "modeled": modeled,
        "unsupported_reason": None if modeled else "instruction semantics are not modeled; effects are conservative/unknown",
        "inputs": row.get("inputs", []),
        "outputs": defs,
        "register_reads": sorted(reads),
        "register_writes": sorted(writes),
        "operand_widths_bits": widths,
        "result_width_bits": row.get("width"),
        "memory_effects": {"reads": memory_reads, "writes": memory_writes,
                           "ordering": row.get("ordering"), "atomic": bool(row.get("atomic")),
                           "unknown": not modeled},
        "flag_effects": flags_effect,
        "control_flow_effects": control,
        "call_effects": ({"clobbers": sorted(writes), "may_read_memory": True,
                          "may_write_memory": True, "target_operand": ins.operands}
                         if opcode == "call" else None),
        "side_effects": bool(memory_writes or flow != "normal" or row.get("atomic")),
    }


def lift_instruction(ins: Instruction, bits: int = 64, block_start: int | None = None) -> dict[str, Any]:
    """Return normalized operation plus explicit register, memory, flag and flow effects."""
    row = _lift_instruction_core(ins, bits=bits, block_start=block_start)
    row["machine_bytes"] = ins.bytes_hex
    row["semantics"] = _semantic_effects(ins, row, bits)
    return row


def lift_blocks(blocks: list[BasicBlock], bits: int = 64) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in sorted(blocks, key=lambda b: b.start):
        for ins in block.instructions:
            row = lift_instruction(ins, bits=bits, block_start=block.start)
            row["successors"] = [hex(x) for x in block.successors]
            rows.append(row)
    return rows
