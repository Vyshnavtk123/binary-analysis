from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import replace

from .api_signatures import API_PARAMETER_TYPES, API_SIGNATURES, normalize_api_name
from .models import Instruction, BasicBlock

# Register families are a small, independent x86/x64 variable model.  They are
# intentionally not a complete CPU semantic specification.
_REGISTER_FAMILIES: dict[str, str] = {}
def _aliases(family: str, *names: str) -> None:
    for name in names:
        _REGISTER_FAMILIES[name] = family

_aliases("rax", "rax", "eax", "ax", "al", "ah")
_aliases("rbx", "rbx", "ebx", "bx", "bl", "bh")
_aliases("rcx", "rcx", "ecx", "cx", "cl", "ch")
_aliases("rdx", "rdx", "edx", "dx", "dl", "dh")
_aliases("rsi", "rsi", "esi", "si", "sil")
_aliases("rdi", "rdi", "edi", "di", "dil")
_aliases("rbp", "rbp", "ebp", "bp", "bpl")
_aliases("rsp", "rsp", "esp", "sp", "spl")
for _n in range(8, 16):
    _aliases(f"r{_n}", f"r{_n}", f"r{_n}d", f"r{_n}w", f"r{_n}b")
_aliases("rsi", "si")
for _n in range(16):
    _aliases(f"xmm{_n}", f"xmm{_n}", f"ymm{_n}", f"zmm{_n}")

_REGISTER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(sorted(_REGISTER_FAMILIES, key=len, reverse=True)) + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CONDITION = {
    "ja": "unsigned_above", "jae": "unsigned_above_or_equal", "jb": "unsigned_below",
    "jbe": "unsigned_below_or_equal", "je": "equal", "jz": "zero",
    "jne": "not_equal", "jnz": "not_zero", "jg": "greater", "jge": "greater_or_equal",
    "jl": "less", "jle": "less_or_equal", "jo": "overflow", "jno": "no_overflow",
    "js": "sign", "jns": "no_sign", "jp": "parity", "jnp": "no_parity",
    "jpe": "parity_even", "jpo": "parity_odd", "jecxz": "ecx_zero", "jrcxz": "rcx_zero",
    "jcxz": "cx_zero", "loop": "loop_condition", "loope": "loop_equal", "loopne": "loop_not_equal",
}
_SIZE_TYPE = {"byte": "uint8_t", "word": "uint16_t", "dword": "uint32_t", "qword": "uint64_t"}
_BINARY = {
    "add": "+", "sub": "-", "imul": "*", "mul": "*", "and": "&", "or": "|",
    "xor": "^", "shl": "<<", "sal": "<<", "shr": ">>", "sar": ">>",
}


def _parse_int(text: str) -> int | None:
    s = text.strip().lower().replace("_", "")
    try:
        if s.startswith("-"):
            return -int(s[1:], 0)
        return int(s, 0)
    except ValueError:
        return None


def _stable_inline_value(value: str | None, immutable_arguments: set[str] | None = None) -> bool:
    """Inline literals and expressions over arguments proven not to be overwritten."""
    if not value or value.startswith("__iat_import__:"):
        return False
    if _parse_int(value) is not None:
        return True
    if re.fullmatch(r"\(uintptr_t\)0x[0-9a-fA-F]+", value):
        return True
    try:
        if isinstance(json.loads(value), str):
            return True
    except (json.JSONDecodeError, TypeError):
        pass
    names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", value))
    if not names or not names.issubset(immutable_arguments or set()):
        return False
    remainder = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", "", value)
    remainder = re.sub(r"0x[0-9a-fA-F]+|\d+", "", remainder)
    return not re.sub(r"[\s()+\-*/&|^<>~]", "", remainder)


class PseudocodeLifter:
    """Independent, conservative x86/x64-to-C-like lifter.

    This stage uses a small variable model and block-local constant propagation.
    Unknown instruction semantics remain visible as comments instead of being
    silently dropped. It does not promise source-equivalent or compilable C.
    """
    def __init__(self, bits: int, strings: dict[int, str] | None = None,
                 imports: dict[int, str] | None = None, function_names: dict[int, str] | None = None):
        self.bits = bits
        self.strings = strings or {}
        self.imports = imports or {}
        self.function_names = function_names or {}
        self.word_type = "uint64_t" if bits == 64 else "uint32_t"
        self.reg_width = 64 if bits == 64 else 32
        # Filled by infer_function_signatures() before pseudocode emission.
        self.function_signatures: dict[int, dict[str, object]] = {}
        self._operand_stack_depths: dict[int, dict[str, int]] = {}
        self._has_stack_depth_context = False
        self._local_types: dict[str, str] = {}
        self._active_register_values: dict[str, str] | None = None
        self._immutable_arguments: set[str] = set()

    def _record_local_type(self, name: str, type_name: str) -> None:
        """Keep the widest observed scalar width for a recovered stack local."""
        rank = {"uint8_t": 1, "uint16_t": 2, "uint32_t": 4, "uint64_t": 8}
        previous = self._local_types.get(name)
        if previous is None or rank.get(type_name, 0) > rank.get(previous, 0):
            self._local_types[name] = type_name

    @staticmethod
    def _split_operands(op_str: str) -> list[str]:
        result: list[str] = []
        buf: list[str] = []
        depth = 0
        for ch in op_str:
            if ch == "[": depth += 1
            elif ch == "]": depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                result.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        if buf:
            result.append("".join(buf).strip())
        return result

    def _argument_register_families(self, blocks: list[BasicBlock]) -> dict[str, str]:
        """Infer Windows x64 register arguments from first-use-before-write.

        This is deliberately a heuristic: indirect effects and complex aliases
        may make the inferred count incomplete or too large.
        """
        if self.bits != 64:
            abi = self._infer_x86_abi(blocks)
            return {str(row["family"]): str(row["name"]) for row in abi["register_parameters"]}
        abi_order = [("rcx", "arg1"), ("rdx", "arg2"), ("r8", "arg3"), ("r9", "arg4")]
        vector_order = [(f"xmm{i}", f"arg{i + 1}") for i in range(4)]
        written: set[str] = set()
        read_before_write: set[str] = set()
        for block in sorted(blocks, key=lambda b: b.start):
            for ins in block.instructions:
                ops = self._split_operands(ins.operands)
                m = ins.mnemonic.lower()
                # Operand zero is a write-only destination for these common forms.
                read_text = ins.operands
                if ops and (m in {"mov", "movabs", "lea", "movzx", "movsx", "movsxd", "pop", "seta", "setae", "setb", "setbe", "sete", "setne", "setg", "setge", "setl", "setle", "setz", "setnz"}
                            or m.startswith(("v", "mov", "cvt", "sqrt", "pextr", "extract"))):
                    read_text = ",".join(ops[1:])
                read_families = { _REGISTER_FAMILIES[x.lower()] for x in _REGISTER_RE.findall(read_text) if x.lower() in _REGISTER_FAMILIES }
                for family, _arg in abi_order:
                    if family in read_families and family not in written:
                        read_before_write.add(family)
                for family, _arg in vector_order:
                    if family in read_families and family not in written:
                        read_before_write.add(family)
                if ops and m not in {"cmp", "test", "push", "call", "jmp", "ret", "nop", "leave"} and not m.startswith("j"):
                    first = ops[0].strip().lower()
                    if first in _REGISTER_FAMILIES:
                        written.add(_REGISTER_FAMILIES[first])
        if not read_before_write:
            return {}
        highest = max(
            i for i in range(4)
            if abi_order[i][0] in read_before_write or vector_order[i][0] in read_before_write
        )
        # Windows x64 uses a single positional register slot: each slot is
        # either the corresponding GPR or XMM register according to argument type.
        # When both are read, retain the GPR candidate and report the ambiguity later.
        result: dict[str, str] = {}
        for index in range(highest + 1):
            gpr, vector = abi_order[index][0], vector_order[index][0]
            family = gpr if gpr in read_before_write else vector if vector in read_before_write else gpr
            result[family] = f"arg{index + 1}"
        return result

    def _pointer_parameter_families(self, blocks: list[BasicBlock],
                                    candidates: set[str]) -> set[str]:
        """Find ABI registers used as address/call targets before a local write."""
        pointer_families: set[str] = set()
        for block in sorted(blocks, key=lambda item: item.start):
            written: set[str] = set()
            for ins in block.instructions:
                operands = self._split_operands(ins.operands)
                for operand in operands:
                    if "[" not in operand:
                        continue
                    for token in _REGISTER_RE.findall(operand):
                        family = _REGISTER_FAMILIES.get(token.lower())
                        if family in candidates and family not in written:
                            pointer_families.add(family)
                if ins.flow == "call" and operands:
                    target = operands[0].strip().lower()
                    family = _REGISTER_FAMILIES.get(target)
                    if family in candidates and family not in written:
                        pointer_families.add(family)
                if operands and ins.mnemonic.lower() not in {
                    "cmp", "test", "push", "call", "callq", "lcall", "jmp",
                    "ret", "retn", "nop", "leave",
                } and not ins.mnemonic.lower().startswith("j"):
                    family = _REGISTER_FAMILIES.get(operands[0].strip().lower())
                    if family:
                        written.add(family)
        return pointer_families

    @staticmethod
    def _x64_vector_argument_type(family: str, blocks: list[BasicBlock]) -> str:
        """Return a conservative scalar/vector type hint from the first XMM use."""
        index = family[3:]
        for block in sorted(blocks, key=lambda item: item.start):
            for ins in block.instructions:
                mnemonic = ins.mnemonic.lower()
                if not re.search(rf"\b(?:xmm|ymm|zmm){index}\b", ins.operands.lower()):
                    continue
                if mnemonic.endswith("ss"):
                    return "float"
                if mnemonic.endswith("sd"):
                    return "double"
                width = 256 if re.search(rf"\bymm{index}\b", ins.operands.lower()) else 512 if re.search(rf"\bzmm{index}\b", ins.operands.lower()) else 128
                return f"vector{width}_t"
        return "vector128_t"

    def _infer_x86_abi(self, blocks: list[BasicBlock], entry: int | None = None) -> dict[str, object]:
        """Infer a conservative x86 stack/register ABI shape from code evidence.

        This is intentionally heuristic. Frame-relative arguments require a
        recognizable EBP setup; ESP-relative arguments are considered only in
        the entry block with local push/pop/sub/add depth adjustment.
        """
        if self.bits != 32:
            return {"calling_convention": "not_x86", "register_parameters": [],
                    "stack_offsets": [], "callee_pop_bytes": None, "warnings": []}
        ordered = sorted(blocks, key=lambda block: block.start)
        entry_block = next((block for block in ordered if block.start == entry), ordered[0] if ordered else None)
        returns = [ins for block in ordered for ins in block.instructions
                   if ins.mnemonic.lower() in {"ret", "retn"}]
        cleanup_values: list[int] = []
        for ins in returns:
            operands = self._split_operands(ins.operands)
            cleanup_values.append(_parse_int(operands[0]) if operands and _parse_int(operands[0]) is not None else 0)
        cleanup = cleanup_values[0] if cleanup_values and len(set(cleanup_values)) == 1 else None

        register_order = [("rcx", "ecx"), ("rdx", "edx")]
        read_before_write: set[str] = set()
        if ordered:
            blocks_by_start = {block.start: block for block in ordered}
            abi_entry = entry if entry in blocks_by_start else ordered[0].start
            reachable = {abi_entry}
            queue = [abi_entry]
            while queue:
                current = queue.pop()
                for successor in blocks_by_start[current].successors:
                    if successor in blocks_by_start and successor not in reachable:
                        reachable.add(successor)
                        queue.append(successor)
            predecessors = {start: set() for start in reachable}
            for start in reachable:
                for successor in blocks_by_start[start].successors:
                    if successor in reachable:
                        predecessors[successor].add(start)

            source_only = {
                "mov", "movabs", "lea", "movzx", "movsx", "movsxd", "pop",
                "seta", "setae", "setb", "setbe", "sete", "setne", "setg",
                "setge", "setl", "setle", "setz", "setnz",
            }
            no_write = {"cmp", "test", "push", "call", "jmp", "ret", "retn", "nop", "leave"}

            def block_writes(start: int) -> set[str]:
                writes: set[str] = set()
                for ins in blocks_by_start[start].instructions:
                    mnemonic = ins.mnemonic.lower()
                    operands = self._split_operands(ins.operands)
                    if (operands and mnemonic not in no_write and not mnemonic.startswith("j")):
                        family = _REGISTER_FAMILIES.get(operands[0].strip().lower())
                        if family in {"rcx", "rdx"}:
                            writes.add(family)
                    if ins.flow in {"conditional", "jump", "return"} or mnemonic.startswith("j"):
                        break
                return writes

            generated = {start: block_writes(start) for start in reachable}
            universe = {family for family, _alias in register_order}
            definitely_in = {start: (set() if start == abi_entry else set(universe)) for start in reachable}
            definitely_out = {start: definitely_in[start] | generated[start] for start in reachable}
            for _ in range(max(1, len(reachable) * 3)):
                changed = False
                for start in sorted(reachable):
                    incoming = [definitely_out[pred] for pred in predecessors[start]]
                    new_in = (set() if start == abi_entry else
                              (set.intersection(*(set(value) for value in incoming)) if incoming else set()))
                    new_out = new_in | generated[start]
                    if new_in != definitely_in[start] or new_out != definitely_out[start]:
                        definitely_in[start], definitely_out[start] = new_in, new_out
                        changed = True
                if not changed:
                    break

            for start in reachable:
                written = set(definitely_in[start])
                for ins in blocks_by_start[start].instructions:
                    operands = self._split_operands(ins.operands)
                    mnemonic = ins.mnemonic.lower()
                    read_text = ins.operands
                    if operands and mnemonic in source_only:
                        read_text = ",".join(operands[1:])
                    if mnemonic in {"xor", "sub"} and len(operands) >= 2 and operands[0].strip().lower() == operands[1].strip().lower():
                        read_text = ",".join(operands[2:])
                    read_families = {
                        _REGISTER_FAMILIES[token.lower()] for token in _REGISTER_RE.findall(read_text)
                        if token.lower() in _REGISTER_FAMILIES
                    }
                    read_before_write.update(
                        family for family, _alias in register_order
                        if family in read_families and family not in written
                    )
                    if operands and mnemonic not in no_write and not mnemonic.startswith("j"):
                        destination = _REGISTER_FAMILIES.get(operands[0].strip().lower())
                        if destination in {"rcx", "rdx"}:
                            written.add(destination)
                    if ins.flow in {"conditional", "jump", "return"} or mnemonic.startswith("j"):
                        break

        # Record EBP's entry-ESP-relative position at a bounded prologue setup.
        # This normalizes both `mov ebp, esp` and equivalent `lea ebp,[esp+N]`
        # frames back to the caller's stack argument coordinates.
        frame_relation: int | None = None
        prologue_depth = 0
        for block in ([entry_block] if entry_block is not None else []):
            for ins in block.instructions:
                mnemonic = ins.mnemonic.lower()
                operands = self._split_operands(ins.operands)
                if mnemonic in {"call", "jmp", "ret", "retn"} or mnemonic.startswith("j"):
                    break
                if len(operands) >= 2 and operands[0].strip().lower() in {"ebp", "bp"}:
                    source = operands[1].strip().lower()
                    if source in {"esp", "sp"} and mnemonic == "mov":
                        frame_relation = -prologue_depth
                        break
                    lea = re.fullmatch(r"\[\s*(?:esp|sp)\s*([+-])?\s*(0x[0-9a-f]+|\d+)?\s*\]", source, re.I)
                    if mnemonic == "lea" and lea:
                        displacement = _parse_int(lea.group(2)) if lea.group(2) else 0
                        if lea.group(1) == "-":
                            displacement = -displacement
                        frame_relation = -prologue_depth + displacement
                        break
                    # An unrecognized EBP definition makes frame coordinates unsafe.
                    frame_relation = None
                    break
                if mnemonic == "push":
                    prologue_depth += 4
                elif mnemonic == "pop":
                    prologue_depth -= 4
                elif mnemonic in {"sub", "add"} and len(operands) >= 2 and operands[0].strip().lower() in {"esp", "sp"}:
                    amount = _parse_int(operands[1])
                    if amount is None:
                        break
                    prologue_depth += amount if mnemonic == "sub" else -amount
                elif operands and operands[0].strip().lower() in {"esp", "sp"}:
                    break
            if frame_relation is not None:
                break
        has_frame_setup = frame_relation is not None
        stack_slots: set[int] = set()

        if has_frame_setup:
            for block in ordered:
                for ins in block.instructions:
                    for operand in self._split_operands(ins.operands):
                        match = re.search(
                            r"(?:(byte|word|dword|qword)\s+ptr\s+)?\[\s*ebp\s*\+\s*(0x[0-9a-f]+|\d+)\s*\]",
                            operand, re.I,
                        )
                        if not match:
                            continue
                        disp = _parse_int(match.group(2))
                        if disp is None:
                            continue
                        disp = disp + frame_relation + 4
                        if disp < 8:
                            continue
                        width = {"byte": 1, "word": 2, "dword": 4, "qword": 8}.get((match.group(1) or "dword").lower(), 4)
                        slot_start = 8 + ((disp - 8) // 4) * 4
                        for offset in range(slot_start, disp + width, 4):
                            stack_slots.add(offset)

        # Without an EBP frame, normalize ESP-relative reads against the entry
        # stack pointer. Propagate exact constant stack depths through CFG edges;
        # blocks with disagreeing or unmodeled incoming depths remain unknown.
        esp_stack_slots: set[int] = set()
        if not has_frame_setup and ordered:
            predecessors: dict[int, list[int]] = {block.start: [] for block in ordered}
            for block in ordered:
                for successor in block.successors:
                    if successor in predecessors:
                        predecessors[successor].append(block.start)
            incoming_depth: dict[int, int | None] = {entry_block.start: 0}
            outgoing_depth: dict[int, int | None] = {}
            per_block_slots: dict[int, set[int]] = {}

            def transfer_stack_depth(block: BasicBlock, initial: int | None) -> tuple[int | None, set[int]]:
                depth = initial
                found: set[int] = set()
                for ins in block.instructions:
                    mnemonic = ins.mnemonic.lower()
                    operands = self._split_operands(ins.operands)
                    read_operands = operands[1:] if mnemonic in {
                        "mov", "movabs", "lea", "movzx", "movsx", "movsxd", "pop",
                    } else operands
                    if depth is not None:
                        for operand in read_operands:
                            match = re.search(
                                r"\[\s*esp\s*(?:([+-])\s*(0x[0-9a-f]+|\d+))?\s*\]",
                                operand, re.I,
                            )
                            if not match:
                                continue
                            displacement = _parse_int(match.group(2)) if match.group(2) else 0
                            if displacement is None:
                                continue
                            if match.group(1) == "-":
                                displacement = -displacement
                            entry_offset = displacement - depth
                            if entry_offset >= 4:
                                slot_start = 4 + ((entry_offset - 4) // 4) * 4
                                width_match = re.search(r"\b(byte|word|dword|qword)\s+ptr\b", operand, re.I)
                                width = {"byte": 1, "word": 2, "dword": 4, "qword": 8}.get(
                                    width_match.group(1).lower() if width_match else "dword", 4
                                )
                                found.update(range(slot_start, entry_offset + width, 4))
                    if ins.flow in {"conditional", "jump", "return"} or mnemonic.startswith("j") or mnemonic in {"ret", "retn", "leave"}:
                        break
                    if mnemonic == "push":
                        depth = None if depth is None else depth + 4
                    elif mnemonic == "pop":
                        depth = None if depth is None else depth - 4
                    elif mnemonic in {"sub", "add"} and len(operands) >= 2 and operands[0].strip().lower() == "esp":
                        amount = _parse_int(operands[1])
                        depth = None if depth is None or amount is None else depth + amount if mnemonic == "sub" else depth - amount
                    elif operands and operands[0].strip().lower() == "esp" and mnemonic not in {"cmp", "test"}:
                        depth = None
                return depth, found

            # Iterate to a fixed point for loops with a consistent stack depth.
            for _ in range(max(1, len(ordered) * 2)):
                changed = False
                for block in ordered:
                    if block.start == entry_block.start:
                        depth_in = 0
                    else:
                        pred_out = [outgoing_depth[p] for p in predecessors[block.start] if p in outgoing_depth]
                        if not predecessors[block.start] or len(pred_out) != len(predecessors[block.start]):
                            continue
                        depth_in = pred_out[0] if all(value == pred_out[0] for value in pred_out) else None
                    if incoming_depth.get(block.start, object()) != depth_in:
                        incoming_depth[block.start] = depth_in
                        changed = True
                    depth_out, found = transfer_stack_depth(block, depth_in)
                    if outgoing_depth.get(block.start, object()) != depth_out:
                        outgoing_depth[block.start] = depth_out
                        changed = True
                    per_block_slots[block.start] = found
                if not changed:
                    break
            for found in per_block_slots.values():
                esp_stack_slots.update(found)

        cleanup_consistent = bool(returns) and cleanup is not None
        cleaned_stack_count = (cleanup + 3) // 4 if cleanup_consistent and cleanup else 0
        if esp_stack_slots:
            # Store in the same normalized coordinate system as EBP arguments:
            # entry ESP + 4 (first argument) maps to EBP + 8.
            stack_slots.update(offset + 4 for offset in esp_stack_slots)
        observed_stack_count = max(((offset - 8) // 4 + 1 for offset in stack_slots), default=0)
        if esp_stack_slots:
            observed_stack_count = max(
                observed_stack_count,
                max(((offset - 4) // 4 + 1 for offset in esp_stack_slots), default=0),
            )
        stack_count = max(cleaned_stack_count, observed_stack_count)
        stack_offsets = [8 + 4 * index for index in range(stack_count)]
        used_ecx = "rcx" in read_before_write
        used_edx = "rdx" in read_before_write

        if used_edx:
            convention = "fastcall (heuristic)"
            register_count = 2
            reg_names = ["arg1", "arg2"]
        elif used_ecx and cleanup_consistent and cleanup and stack_count:
            convention = "thiscall candidate (heuristic; fastcall is also possible)"
            register_count = 1
            reg_names = ["this"]
        elif used_ecx:
            convention = "x86 register-call candidate (thiscall/fastcall ambiguous)"
            register_count = 1
            reg_names = ["arg1"]
        elif cleanup_consistent and cleanup:
            convention = "stdcall (heuristic)"
            register_count = 0
            reg_names = []
        elif stack_count:
            convention = "cdecl candidate (heuristic)"
            register_count = 0
            reg_names = []
        else:
            convention = "x86 convention unknown"
            register_count = 0
            reg_names = []

        register_families = ["rcx", "rdx"][:register_count]
        register_parameters = [
            {"name": name, "family": family, "register": "ecx" if family == "rcx" else "edx"}
            for name, family in zip(reg_names, register_families)
        ]
        warnings: list[str] = []
        if returns and cleanup is None:
            warnings.append("Return instructions disagree on callee stack cleanup.")
        if cleanup_consistent and cleanup and cleanup % 4:
            warnings.append("Callee stack cleanup is not a multiple of the x86 word size.")
        if cleanup_consistent and cleanup and observed_stack_count > cleaned_stack_count:
            warnings.append("Observed stack arguments exceed ret-immediate cleanup evidence.")
        if not has_frame_setup and any("[esp" in ins.operands.lower() for block in ordered for ins in block.instructions) and not esp_stack_slots:
            warnings.append("ESP-relative argument offsets are not normalized across the function stack frame.")
        return {
            "calling_convention": convention,
            "callee_pop_bytes": cleanup,
            "register_parameters": register_parameters,
            "register_parameter_count": register_count,
            "stack_offsets": stack_offsets,
            "stack_parameter_count": stack_count,
            "return_cleanup_evidence": cleanup_values,
            "register_argument_evidence": sorted(read_before_write),
            "stack_argument_evidence": sorted(stack_slots),
            "esp_stack_argument_evidence": sorted(esp_stack_slots),
            "warnings": warnings,
        }

    _CALL_ABI = ("rcx", "rdx", "r8", "r9")
    _CALLER_SAVED = ("rax", "rcx", "rdx", "r8", "r9", "r10", "r11")
    _IMPORT_ARITY = {
        name: int(signature["arity"])
        for name, signature in API_SIGNATURES.items()
        if signature["arity"] is not None
    }
    _IMPORT_RETURN = {
        name: str(signature["return_type"])
        for name, signature in API_SIGNATURES.items()
        if signature["return_type"] is not None
    }

    @staticmethod
    def _base_symbol(name: str | None) -> str:
        return normalize_api_name(name)

    def _import_name_for(self, ins: Instruction) -> str | None:
        if ins.import_name:
            return ins.import_name
        if ins.reference_target in self.imports:
            return self.imports[ins.reference_target]
        return None

    @classmethod
    def _type_import_arguments(cls, import_name: str | None, arguments: list[str]) -> list[str]:
        """Apply only known pointer casts from the bounded import catalog."""
        types = API_PARAMETER_TYPES.get(normalize_api_name(import_name), ())
        typed = list(arguments)
        for index, type_name in enumerate(types[:len(typed)]):
            if "*" in type_name:
                typed[index] = f"({type_name})({typed[index]})"
        return typed

    def _written_register_family(self, ins: Instruction) -> tuple[str, str] | None:
        """Return (register family, textual alias) for a recognized register destination."""
        m = ins.mnemonic.lower()
        ops = self._split_operands(ins.operands)
        if not ops or m in {"cmp", "test", "push", "call", "callq", "lcall", "jmp", "ret", "retq", "retn", "nop", "leave"} or m.startswith("j"):
            return None
        first = ops[0].strip().lower()
        family = _REGISTER_FAMILIES.get(first)
        if not family:
            return None
        return family, first

    def _return_def_width(self, ins: Instruction, known: dict[int, dict[str, object]]) -> int | str | None:
        """Return a width or type hint when an instruction defines EAX/RAX."""
        m = ins.mnemonic.lower()
        if m in {"call", "callq", "lcall"}:
            if ins.direct_target is not None and ins.direct_target in known:
                ret_type = str(known[ins.direct_target].get("return_type", "void"))
                if ret_type in {"uintptr_t", "size_t", "bool", "int32_t"}:
                    return ret_type
                if ret_type == "uint32_t":
                    return 32
                if ret_type == "uint64_t":
                    return 64
            imported = self._base_symbol(self._import_name_for(ins))
            ret_type = self._IMPORT_RETURN.get(imported)
            if ret_type in {"uintptr_t", "size_t", "bool", "int32_t"}:
                return ret_type
            if ret_type == "uint32_t":
                return 32
            if ret_type == "uint64_t":
                return 64
            return None
        written = self._written_register_family(ins)
        if not written or written[0] != "rax":
            return None
        alias = written[1]
        if alias in {"rax"}:
            return 64
        if alias in {"eax", "ax", "al", "ah"}:
            # A 32-bit write zero-extends EAX into RAX on x86-64. Smaller writes
            # are promoted to a conservative 32-bit return classification.
            return 32
        return 32

    def _infer_return_signature(self, blocks: list[BasicBlock], known: dict[int, dict[str, object]], entry_address: int | None = None) -> tuple[str, str, int, int]:
        """Use a bounded must-reach-definition pass for values returned in EAX/RAX."""
        if not blocks:
            return "void", "none", 0, 0
        by_start = {b.start: b for b in blocks}
        starts = set(by_start)
        preds: dict[int, set[int]] = {start: set() for start in starts}
        for block in blocks:
            for succ in block.successors:
                if succ in starts:
                    preds[succ].add(block.start)
        entry = entry_address if entry_address in starts else min(starts)
        # State is the set of possible last definitions: 0 means no recognized
        # return-register definition has reached this point.
        in_state: dict[int, set[int | str]] = {start: {0} for start in starts}
        out_state: dict[int, set[int | str]] = {start: {0} for start in starts}
        def transfer(block: BasicBlock, state_in: set[int | str]) -> set[int | str]:
            state = set(state_in)
            for ins in block.instructions:
                width = self._return_def_width(ins, known)
                if width is not None:
                    state = {width}
            return state
        for _ in range(max(8, len(blocks) * 4)):
            changed = False
            for start in sorted(starts):
                if start == entry:
                    new_in = {0}
                else:
                    incoming = preds[start]
                    new_in = set().union(*(out_state[pred] for pred in incoming)) if incoming else {0}
                new_out = transfer(by_start[start], new_in)
                if new_in != in_state[start] or new_out != out_state[start]:
                    in_state[start], out_state[start] = new_in, new_out
                    changed = True
            if not changed:
                break
        returns: list[set[int]] = []
        for start in sorted(starts):
            state = set(in_state[start])
            for ins in by_start[start].instructions:
                if ins.mnemonic.lower() in {"ret", "retq", "retn", "retf", "iret", "iretq"}:
                    returns.append(set(state))
                    break
                width = self._return_def_width(ins, known)
                if width is not None:
                    state = {width}
        if not returns:
            # A tail jump to a known non-void function is commonly a forwarding return.
            for block in blocks:
                if block.instructions:
                    last = block.instructions[-1]
                    tail_target = (
                        last.direct_target if last.direct_target is not None
                        else last.resolved_indirect_jump_target
                    )
                    if last.flow == "jump" and tail_target in known:
                        ret_type = str(known[tail_target].get("return_type", "void"))
                        if ret_type in {"uint32_t", "uint64_t", "uintptr_t", "size_t", "bool", "int32_t"}:
                            return ret_type, "tail-call", 0, 0
            return "void", "none", 0, 0
        value_paths = sum(1 for state in returns if state and 0 not in state)
        if value_paths != len(returns):
            return "void", "none", len(returns), value_paths
        widths = set().union(*returns)
        if len(widths) == 1 and isinstance(next(iter(widths)), str) and next(iter(widths)) != 0:
            ret_type = next(iter(widths))
            reg = "rax" if self.bits == 64 and ret_type in {"uintptr_t", "size_t"} else "eax"
        elif widths.issubset({32, 64}):
            ret_type = "uint64_t" if 64 in widths else "uint32_t"
            reg = "rax" if ret_type == "uint64_t" else "eax"
        else:
            return "void", "none", len(returns), value_paths
        return ret_type, reg, len(returns), value_paths

    def infer_function_signatures(self, function_blocks: dict[int, list[BasicBlock]]) -> list[dict[str, object]]:
        """Infer parameter counts and scalar return types before emitting function text.

        This is deliberately conservative and heuristic. Windows x64 stack arguments
        are recovered only from positive frame offsets or entry-block RSP offsets
        that can be normalized across the observed prologue.
        """
        signatures: dict[int, dict[str, object]] = {}
        for entry, blocks in function_blocks.items():
            arg_families = self._argument_register_families(blocks)
            pointer_parameters = self._pointer_parameter_families(
                blocks, set(arg_families)
            )
            if self.bits == 64:
                count = max((int(name[3:]) for name in arg_families.values()), default=0)
                stack_offsets: set[int] = set()
                ordered_blocks = sorted(blocks, key=lambda block: block.start)
                entry_block = next((block for block in ordered_blocks if block.start == entry),
                                   ordered_blocks[0] if ordered_blocks else None)
                rbp_depth: int | None = None
                rsp_depth_at: dict[int, int | None] = {}
                if entry_block is not None:
                    depth = 0
                    for ins in entry_block.instructions:
                        operands = self._split_operands(ins.operands)
                        mnemonic = ins.mnemonic.lower()
                        if mnemonic in {"call", "jmp", "ret", "retn"} or mnemonic.startswith("j"):
                            break
                        if len(operands) >= 2 and operands[0].strip().lower() == "rbp":
                            if mnemonic == "mov" and operands[1].strip().lower() == "rsp":
                                rbp_depth = depth
                            elif mnemonic == "lea":
                                frame = re.fullmatch(
                                    r"\[\s*rsp\s*([+-])?\s*(0x[0-9a-f]+|\d+)?\s*\]",
                                    operands[1].strip(), re.I,
                                )
                                if frame:
                                    displacement = _parse_int(frame.group(2)) if frame.group(2) else 0
                                    if frame.group(1) == "-":
                                        displacement = -displacement
                                    rbp_depth = depth - displacement
                        if mnemonic == "push":
                            depth += 8
                        elif mnemonic == "pop":
                            depth -= 8
                        elif mnemonic in {"sub", "add"} and len(operands) >= 2 and operands[0].strip().lower() == "rsp":
                            amount = _parse_int(operands[1])
                            if amount is None:
                                break
                            depth += amount if mnemonic == "sub" else -amount
                        elif operands and operands[0].strip().lower() == "rsp":
                            break
                if entry_block is not None:
                    blocks_by_start = {block.start: block for block in ordered_blocks}
                    predecessors = {start: set() for start in blocks_by_start}
                    for block in ordered_blocks:
                        for successor in block.successors:
                            if successor in predecessors:
                                predecessors[successor].add(block.start)
                    incoming_depth: dict[int, int | None] = {entry_block.start: 0}
                    outgoing_depth: dict[int, int | None] = {}
                    per_block_depths: dict[int, dict[int, int | None]] = {}

                    def rsp_transfer(block: BasicBlock, initial: int | None) -> tuple[int | None, dict[int, int | None]]:
                        current_depth = initial
                        at_instruction: dict[int, int | None] = {}
                        for current_ins in block.instructions:
                            at_instruction[current_ins.address] = current_depth
                            current_mnemonic = current_ins.mnemonic.lower()
                            current_operands = self._split_operands(current_ins.operands)
                            if (current_ins.flow in {"conditional", "jump", "return"}
                                    or current_mnemonic.startswith("j")
                                    or current_mnemonic in {"ret", "retn", "leave"}):
                                if current_mnemonic == "leave":
                                    current_depth = None
                                break
                            if current_mnemonic == "push":
                                current_depth = None if current_depth is None else current_depth + 8
                            elif current_mnemonic == "pop":
                                current_depth = None if current_depth is None else current_depth - 8
                            elif (current_mnemonic in {"sub", "add"} and len(current_operands) >= 2
                                  and current_operands[0].strip().lower() == "rsp"):
                                amount = _parse_int(current_operands[1])
                                if amount is None or current_depth is None:
                                    current_depth = None
                                else:
                                    current_depth += amount if current_mnemonic == "sub" else -amount
                            elif (current_mnemonic == "lea" and len(current_operands) >= 2
                                  and current_operands[0].strip().lower() == "rsp"):
                                adjustment = re.fullmatch(
                                    r"\[\s*rsp\s*([+-])\s*(0x[0-9a-f]+|\d+)\s*\]",
                                    current_operands[1].strip(), re.I,
                                )
                                amount = _parse_int(adjustment.group(2)) if adjustment else None
                                if amount is None or current_depth is None:
                                    current_depth = None
                                else:
                                    current_depth += -amount if adjustment.group(1) == "+" else amount
                            elif (current_operands and current_operands[0].strip().lower() == "rsp"
                                  and current_mnemonic not in {"cmp", "test"}):
                                current_depth = None
                            if current_depth is not None and abs(current_depth) > 0x100000:
                                current_depth = None
                        return current_depth, at_instruction

                    for _ in range(max(1, len(ordered_blocks) * 3)):
                        changed = False
                        for current_block in ordered_blocks:
                            start = current_block.start
                            if start == entry_block.start:
                                depth_in: int | None = 0
                            else:
                                preds = predecessors[start]
                                pred_depths = [outgoing_depth[pred] for pred in preds if pred in outgoing_depth]
                                if not preds or not pred_depths:
                                    continue
                                depth_in = pred_depths[0] if all(value == pred_depths[0] for value in pred_depths) else None
                            if incoming_depth.get(start, object()) != depth_in:
                                incoming_depth[start] = depth_in
                                changed = True
                            depth_out, at_instruction = rsp_transfer(current_block, depth_in)
                            if outgoing_depth.get(start, object()) != depth_out:
                                outgoing_depth[start] = depth_out
                                changed = True
                            per_block_depths[start] = at_instruction
                        if not changed:
                            break
                    for at_instruction in per_block_depths.values():
                        rsp_depth_at.update(at_instruction)
                for block in blocks:
                    for ins in block.instructions:
                        for operand in self._split_operands(ins.operands):
                            match = re.search(r"\[(rbp|ebp|rsp|esp)\s*\+\s*(0x[0-9a-f]+|\d+)\]", operand, re.I)
                            if match:
                                offset = _parse_int(match.group(2))
                                base = match.group(1).lower()
                                if base in {"rbp", "ebp"}:
                                    if rbp_depth is None:
                                        continue
                                    offset = offset - rbp_depth if offset is not None else None
                                else:
                                    depth = rsp_depth_at.get(ins.address)
                                    offset = offset - depth if offset is not None and depth is not None else None
                                min_argument_offset = 0x28
                                if offset is not None and offset >= min_argument_offset:
                                    stack_offsets.add(offset)
                stack_parameters = [f"stack_arg_{offset:X}" for offset in sorted(stack_offsets)]
                register_parameters = []
                for family, name in sorted(arg_families.items(), key=lambda row: int(row[1][3:])):
                    row: dict[str, object] = {
                        "name": name, "register": family,
                        "position": int(name[3:]) - 1,
                    }
                    if family.startswith("xmm"):
                        row["type"] = self._x64_vector_argument_type(family, blocks)
                    else:
                        row["type"] = "uintptr_t" if family in pointer_parameters else self.word_type
                    register_parameters.append(row)
                calling_convention = "Windows x64 register + shadow-space arguments (heuristic)"
                callee_pop_bytes = None
                abi_evidence: dict[str, object] = {
                    "vector_register_candidates": [
                        str(row["register"]) for row in register_parameters
                        if str(row["register"]).startswith("xmm")
                    ],
                    "pointer_register_candidates": [
                        str(row["register"]) for row in register_parameters
                        if row.get("type") == "uintptr_t"
                    ]
                }
                abi_warnings: list[str] = []
            else:
                x86_abi = self._infer_x86_abi(blocks, entry)
                register_parameters = []
                for row in x86_abi["register_parameters"]:
                    family = str(row["family"])
                    register_parameters.append({
                        "name": str(row["name"]),
                        "register": str(row["register"]),
                        "type": "uintptr_t" if family in pointer_parameters else self.word_type,
                    })
                count = int(x86_abi["register_parameter_count"]) + int(x86_abi["stack_parameter_count"])
                stack_parameters = [f"stack_arg_{offset:X}" for offset in x86_abi["stack_offsets"]]
                calling_convention = str(x86_abi["calling_convention"])
                callee_pop_bytes = x86_abi["callee_pop_bytes"]
                abi_evidence = {
                    "return_cleanup_bytes": x86_abi["return_cleanup_evidence"],
                    "register_arguments_read_before_write": x86_abi["register_argument_evidence"],
                    "frame_argument_offsets_observed": [hex(x) for x in x86_abi["stack_argument_evidence"]],
                    "esp_argument_offsets_observed": [hex(x) for x in x86_abi["esp_stack_argument_evidence"]],
                }
                abi_warnings = list(x86_abi["warnings"])
            parameter_names = [str(row["name"]) for row in register_parameters] + stack_parameters
            signatures[entry] = {
                "address": hex(entry),
                "name": self.function_names.get(entry, f"sub_{entry:X}"),
                "parameter_count": len(parameter_names),
                "register_parameter_count": len(register_parameters),
                "parameters": parameter_names,
                "register_parameters": register_parameters,
                "stack_parameters": stack_parameters,
                "shadow_space_bytes": 32 if self.bits == 64 else 0,
                "stack_alignment_bytes": 16 if self.bits == 64 else 4,
                "calling_convention": calling_convention,
                "calling_convention_confidence": "heuristic",
                "callee_pop_bytes": callee_pop_bytes,
                "abi_evidence": abi_evidence,
                "abi_warnings": abi_warnings,
                "return_type": "void",
                "return_register": "none",
                "return_paths": 0,
                "return_paths_with_value": 0,
                "return_analysis": "not yet analyzed",
            }
        # Iterate so calls and proven tail jumps to functions already inferred as
        # returning a scalar can propagate that return type to their callers.
        for _ in range(max(2, min(8, len(signatures) + 1))):
            changed = False
            for entry, blocks in function_blocks.items():
                ret_type, ret_reg, return_paths, value_paths = self._infer_return_signature(blocks, signatures, entry)
                rec = signatures[entry]
                new_values = (ret_type, ret_reg, return_paths, value_paths)
                old_values = (rec["return_type"], rec["return_register"], rec["return_paths"], rec["return_paths_with_value"])
                if new_values != old_values:
                    rec["return_type"], rec["return_register"], rec["return_paths"], rec["return_paths_with_value"] = new_values
                    if ret_type == "void":
                        rec["return_analysis"] = "no reliable scalar value reaches every recovered return"
                    elif ret_reg == "tail-call":
                        rec["return_analysis"] = "return type forwarded from a known tail-call target"
                    else:
                        rec["return_analysis"] = "EAX/RAX definition reaches every recovered return"
                    changed = True
            if not changed:
                break
        self.function_signatures = signatures
        return [dict(signatures[e]) for e in sorted(signatures)]

    def _is_rax_read_after_call(self, block: BasicBlock, instruction_index: int, caller_returns_value: bool) -> bool:
        """Find whether the call result is read before a new RAX definition/clobber."""
        for later in block.instructions[instruction_index + 1:]:
            m = later.mnemonic.lower()
            if m in {"call", "callq", "lcall"}:
                return False
            if m in {"ret", "retq", "retn", "retf", "iret", "iretq"}:
                return caller_returns_value
            ops = self._split_operands(later.operands)
            if not ops:
                continue
            # These instructions do not read their first operand as a source.
            if m in {"mov", "movabs", "lea", "movzx", "movsx", "movsxd", "pop", "seta", "setae", "setb", "setbe", "sete", "setne", "setg", "setge", "setl", "setle", "setz", "setnz"}:
                read_text = ",".join(ops[1:])
            elif m == "xor" and len(ops) >= 2 and ops[0].strip().lower() == ops[1].strip().lower():
                read_text = ",".join(ops[2:])
            else:
                read_text = later.operands
            if any(_REGISTER_FAMILIES.get(token.lower()) == "rax" for token in _REGISTER_RE.findall(read_text)):
                return True
            # A write to RAX before a read kills the call's result.
            write = self._written_register_family(later)
            if write and write[0] == "rax":
                return False
        return False

    def _value_for_call_operand(self, operand: str, ins: Instruction,
                                argument_families: dict[str, str], constants: dict[str, int],
                                register_values: dict[str, str]) -> str:
        value = operand.strip()
        family = _REGISTER_FAMILIES.get(value.lower())
        if family:
            if family in constants:
                return str(constants[family])
            return register_values.get(family, self._register_name(value, argument_families))
        immediate = _parse_int(value)
        if immediate is not None:
            return str(immediate)
        return self._operand(value, ins, argument_families, constants)

    def _call_arguments(self, ins: Instruction, argument_families: dict[str, str],
                        constants: dict[str, int], register_values: dict[str, str],
                        written_since_call: set[str],
                        pending_stack_arguments: list[str | None] | None = None,
                        outgoing_stack_values: dict[int, str] | None = None,
                        stack_depth: int = 0) -> list[str]:
        abi = self._CALL_ABI if self.bits == 64 else ()
        target_signature = self.function_signatures.get(ins.direct_target) if ins.direct_target is not None else None
        import_name = self._import_name_for(ins)
        pending_stack_arguments = pending_stack_arguments or []
        if self.bits == 32:
            if target_signature is not None:
                args: list[str] = []
                for row in target_signature.get("register_parameters", []):
                    if not isinstance(row, dict):
                        continue
                    register = str(row.get("register", "")).lower()
                    family = _REGISTER_FAMILIES.get(register)
                    if not family:
                        continue
                    value = str(constants[family]) if family in constants else register_values.get(family)
                    if value is None and family in argument_families and family not in written_since_call:
                        value = argument_families[family]
                    argument = value or f"v_{family}"
                    if row.get("type") == "uintptr_t":
                        argument = f"(uintptr_t)({argument})"
                    args.append(argument)
                stack_count = len(target_signature.get("stack_parameters", []))
                if stack_count:
                    push_values = list(reversed(pending_stack_arguments[-stack_count:]))
                    missing = stack_count - len(push_values)
                    push_values = [None] * max(0, missing) + push_values
                    outgoing_stack_values = outgoing_stack_values or {}
                    ordered_parameters = sorted(
                        (str(parameter) for parameter in target_signature.get("stack_parameters", [])),
                        key=lambda parameter: int(parameter.rsplit("_", 1)[-1], 16),
                    )
                    for index, parameter in enumerate(ordered_parameters, start=1):
                        callee_offset = int(parameter.rsplit("_", 1)[-1], 16)
                        caller_key = callee_offset - 8 - stack_depth
                        # Keep the established right-to-left push interpretation
                        # when pushes are present; mixed pushes and direct ESP
                        # stores do not have enough ordering evidence to combine.
                        value = (outgoing_stack_values.get(caller_key)
                                 if not pending_stack_arguments else None)
                        if value in {None, "?"}:
                            value = push_values[index - 1] if index <= len(push_values) else None
                        args.append(value if value is not None else f"unknown_stack_arg_{index}")
                return args
            import_arity = self._IMPORT_ARITY.get(self._base_symbol(import_name))
            if import_arity is not None and import_arity > 0:
                available_values = list(reversed(pending_stack_arguments[-import_arity:]))
                missing = import_arity - len(available_values)
                args = [f"unknown_stack_arg_{index}" for index in range(1, missing + 1)]
                for index, value in enumerate(available_values, start=missing + 1):
                    args.append(value if value is not None else f"unknown_stack_arg_{index}")
                return self._type_import_arguments(import_name, args)
            return []
        args: list[str] = []
        if target_signature is not None:
            rows = [row for row in target_signature.get("register_parameters", []) if isinstance(row, dict)]
            rows.sort(key=lambda row: int(row.get("position", str(row.get("name", "arg1"))[3:]) or 0))
            for row in rows:
                register = str(row.get("register", "")).lower()
                family = _REGISTER_FAMILIES.get(register)
                if family is None:
                    args.append(f"unknown_{row.get('name', 'arg')}")
                    continue
                arg = str(constants[family]) if family in constants else register_values.get(family)
                if family in argument_families and family not in written_since_call:
                    arg = argument_families[family]
                fallback = f"v_{family}" if family in self._CALL_ABI else f"unknown_{row.get('name', family)}"
                argument = arg or fallback
                if target_signature is not None and row.get("type") == "uintptr_t":
                    argument = f"(uintptr_t)({argument})"
                args.append(argument)
        else:
            import_arity = self._IMPORT_ARITY.get(self._base_symbol(import_name))
            if import_arity is not None:
                count = import_arity
            else:
                slot_families = [
                    (self._CALL_ABI[index], f"xmm{index}") for index in range(4)
                ]
                highest = max((index for index, (gpr, xmm) in enumerate(slot_families)
                               if gpr in written_since_call or xmm in written_since_call), default=-1)
                count = highest + 1
            for index in range(min(count, 4)):
                gpr, xmm = self._CALL_ABI[index], f"xmm{index}"
                family = gpr if gpr in written_since_call or gpr in argument_families else xmm
                arg = register_values.get(family)
                if family in argument_families and family not in written_since_call:
                    arg = argument_families[family]
                args.append(arg or f"v_{family}")
            if target_signature is None:
                args = self._type_import_arguments(import_name, args)
        if self.bits == 64 and target_signature is not None:
            outgoing_stack_values = outgoing_stack_values or {}
            stack_parameters = sorted(
                (str(param) for param in target_signature.get("stack_parameters", [])),
                key=lambda param: int(param.rsplit("_", 1)[-1], 16),
            )
            for index, parameter in enumerate(stack_parameters, start=5):
                callee_offset = int(parameter.rsplit("_", 1)[-1], 16)
                caller_key = callee_offset - 8 - stack_depth
                args.append(outgoing_stack_values.get(caller_key, f"unknown_stack_arg_{index}"))
        return args

    def _update_register_values(self, ins: Instruction, argument_families: dict[str, str],
                                constants: dict[str, int], register_values: dict[str, str]) -> None:
        """Maintain simple symbolic register values for call argument reconstruction."""
        m = ins.mnemonic.lower()
        written = self._written_register_family(ins)
        if not written:
            return
        family, alias = written
        ops = self._split_operands(ins.operands)
        if not ops:
            return
        old_value = register_values.get(family, f"v_{family}")
        if m in {
            "mov", "movabs", "movzx", "movsx", "movsxd", "movd", "vmovd",
            "movq", "vmovq", "movss", "vmovss", "movsd", "vmovsd",
            "movaps", "vmovaps", "movups", "vmovups", "movapd", "vmovapd",
            "movupd", "vmovupd", "movdqa", "vmovdqa", "movdqu", "vmovdqu",
        } and len(ops) >= 2:
            src = ops[1].strip()
            if m == "mov" and "[" in src and ins.reference_target in self.imports:
                value = f"__iat_import__:{self.imports[ins.reference_target]}"
            else:
                immediate = _parse_int(src)
                if immediate is not None:
                    value = str(immediate)
                else:
                    src_family = _REGISTER_FAMILIES.get(src.lower())
                    if src_family:
                        value = str(constants[src_family]) if src_family in constants else register_values.get(src_family, self._register_name(src, argument_families))
                    else:
                        value = self._operand(src, ins, argument_families, constants)
            register_values[family] = value
            return
        if m == "lea" and len(ops) >= 2:
            if ins.reference_target in self.strings:
                register_values[family] = json.dumps(self.strings[ins.reference_target], ensure_ascii=False)
            elif ins.reference_target is not None:
                register_values[family] = f"(uintptr_t)0x{ins.reference_target:X}"
            else:
                register_values[family] = f"address_of({self._operand(ops[1], ins, argument_families, constants)})"
            return
        if m in _BINARY and len(ops) >= 2:
            if m == "xor" and ops[0].strip().lower() == ops[1].strip().lower():
                register_values[family] = "0"
                return
            lhs = old_value if len(ops) == 2 else self._value_for_call_operand(ops[1], ins, argument_families, constants, register_values)
            rhs_text = ops[-1].strip()
            rhs_number = _parse_int(rhs_text)
            rhs_family = _REGISTER_FAMILIES.get(rhs_text.lower())
            rhs = str(constants[rhs_family]) if rhs_family in constants else register_values.get(rhs_family, self._register_name(rhs_text, argument_families)) if rhs_family else self._operand(rhs_text, ins, argument_families, constants)
            left_number = _parse_int(lhs)
            right_number = rhs_number if rhs_number is not None else _parse_int(rhs)
            if left_number is not None and right_number is not None:
                folded = self._const_eval(m, left_number, right_number)
                register_values[family] = str(folded) if folded is not None else f"({lhs} {_BINARY[m]} {rhs})"
            else:
                register_values[family] = f"({lhs} {_BINARY[m]} {rhs})"
            return
        if m in {"inc", "dec", "neg", "not"}:
            if m == "inc": register_values[family] = f"({old_value} + 1)"
            elif m == "dec": register_values[family] = f"({old_value} - 1)"
            elif m == "neg": register_values[family] = f"(-({old_value}))"
            else: register_values[family] = f"(~({old_value}))"
            return
        register_values[family] = self._register_name(alias, {})

    def _register_name(self, token: str, argument_families: dict[str, str]) -> str:
        low = token.lower()
        family = _REGISTER_FAMILIES.get(low)
        if family is None:
            return token
        if family in argument_families:
            return argument_families[family]
        # Keep stack/frame pointers explicit because they are needed to understand
        # stack accesses; general-purpose registers become stable pseudo-variables.
        canonical = family
        prefix = "v_" if canonical not in {"rsp", "rbp"} else ""
        return prefix + canonical

    def _memory_expr(self, text: str, argument_families: dict[str, str],
                     ins: Instruction | None = None) -> str:
        m = re.match(r"\s*(byte|word|dword|qword)\s+ptr\s+\[(.+)\]\s*$", text, re.I)
        if not m:
            return text.strip()
        size, address = m.group(1).lower(), m.group(2).strip()
        typ = _SIZE_TYPE[size]
        normalized = address.lower().replace(" ", "")
        # Common frame-relative slots are promoted to local names instead of
        # rendering unreadable *(uint32_t *)(rbp - 0x4) expressions.
        stack_match = re.fullmatch(r"(?:rbp|ebp|rsp|esp)([+-])(0x[0-9a-f]+|\d+)", normalized)
        if stack_match:
            sign, offset_text = stack_match.groups()
            offset = int(offset_text, 0)
            base_match = re.match(r"(rbp|ebp|rsp|esp)", normalized)
            base = base_match.group(1) if base_match else ""
            depth = self._operand_stack_depths.get(ins.address, {}).get(base) if ins else None
            if depth is not None and self.bits == 64:
                displacement = offset if sign == "+" else -offset
                entry_offset = displacement - depth
                if entry_offset >= 0x28:
                    return f"stack_arg_{entry_offset:X}"
                local_name = f"local_{abs(entry_offset):X}"
                self._record_local_type(local_name, typ)
                return local_name
            elif self.bits == 64 and ins is not None and self._has_stack_depth_context:
                address = _REGISTER_RE.sub(
                    lambda match: self._register_name(match.group(1), argument_families), address
                )
                return f"*((({typ} *)({address})))"
            if sign == "-":
                local_name = f"local_{offset:X}"
                self._record_local_type(local_name, typ)
                return local_name
            return f"stack_arg_{offset:X}"
        # Apply register naming to the address expression.
        def address_register(match: re.Match[str]) -> str:
            token = match.group(1)
            family = _REGISTER_FAMILIES.get(token.lower())
            if self._active_register_values is not None and family:
                value = self._active_register_values.get(family)
                if _stable_inline_value(value, self._immutable_arguments):
                    return f"({value})"
            return self._register_name(token, argument_families)
        address = _REGISTER_RE.sub(address_register, address)
        return f"*((({typ} *)({address})))"

    def _operand(self, text: str, ins: Instruction | None = None,
                 argument_families: dict[str, str] | None = None,
                 constants: dict[str, int] | None = None, destination: bool = False) -> str:
        argument_families = argument_families or {}
        constants = constants or {}
        value = text.strip()
        # Handle memory operands first so register substitutions do not corrupt
        # stack-address pattern detection.
        memory = self._memory_expr(value, argument_families, ins)
        if memory != value:
            value = memory
        else:
            def replace_register(match: re.Match[str]) -> str:
                token = match.group(1)
                family = _REGISTER_FAMILIES.get(token.lower())
                if family and not destination:
                    if family in constants:
                        return str(constants[family])
                    if self._active_register_values is not None:
                        value = self._active_register_values.get(family)
                        if _stable_inline_value(value, self._immutable_arguments):
                            return f"({value})"
                return self._register_name(token, argument_families)
            value = _REGISTER_RE.sub(replace_register, value)
        if ins and "rip" in text.lower() and ins.reference_target is not None:
            if "ptr" in value and value.startswith("*((("):
                return value
            return f"(uintptr_t)0x{ins.reference_target:X}"
        return value

    @staticmethod
    def _const_eval(mnemonic: str, left: int, right: int) -> int | None:
        try:
            if mnemonic == "add": return left + right
            if mnemonic == "sub": return left - right
            if mnemonic in {"imul", "mul"}: return left * right
            if mnemonic == "and": return left & right
            if mnemonic == "or": return left | right
            if mnemonic == "xor": return left ^ right
            if mnemonic in {"shl", "sal"}: return left << max(0, right)
            if mnemonic in {"shr", "sar"}: return left >> max(0, right)
        except (OverflowError, ValueError):
            return None
        return None

    def _condition_expression(self, branch: Instruction, flag_source: Instruction | None,
                              arg_families: dict[str, str] | None = None,
                              constants: dict[str, int] | None = None) -> str | None:
        arg_families = arg_families or {}
        constants = constants or {}
        if flag_source is None:
            return None
        bm = branch.mnemonic.lower()
        sm = flag_source.mnemonic.lower()
        operands = self._split_operands(flag_source.operands)
        if sm == "cmp" and len(operands) >= 2:
            left = self._operand(operands[0], flag_source, arg_families, constants)
            right = self._operand(operands[1], flag_source, arg_families, constants)
            relation = {
                "je": "==", "jz": "==", "jne": "!=", "jnz": "!=", "jg": ">", "jnle": ">",
                "jge": ">=", "jnl": ">=", "jl": "<", "jnge": "<", "jle": "<=", "jng": "<=",
                "ja": ">", "jnbe": ">", "jae": ">=", "jnb": ">=", "jnc": ">=",
                "jb": "<", "jnae": "<", "jc": "<", "jbe": "<=", "jna": "<=",
            }.get(bm)
            if relation:
                unsigned = bm in {"ja", "jnbe", "jae", "jnb", "jnc", "jb", "jnae", "jc", "jbe", "jna"}
                return f"((({self.word_type})({left})) {relation} (({self.word_type})({right})))" if unsigned else f"({left} {relation} {right})"
        if sm == "test" and operands:
            left = self._operand(operands[0], flag_source, arg_families, constants)
            right = self._operand(operands[1], flag_source, arg_families, constants) if len(operands) > 1 else left
            if bm in {"je", "jz"}: return f"(({left} & {right}) == 0)"
            if bm in {"jne", "jnz"}: return f"(({left} & {right}) != 0)"
        return None

    def _format_dataflow_condition(self, expression: str, ins: Instruction,
                                   argument_families: dict[str, str],
                                   constants: dict[str, int],
                                   flag_source: Instruction | None) -> str | None:
        """Render simple path-sensitive data-flow branch conditions as C-like text."""
        value = expression.strip()
        if value in {"0", "1"}:
            return value
        # Memory expressions need access-width and alias information at the read
        # site. Keep the existing instruction-based condition renderer for them.
        if re.search(r"\bmem\s*\[", value):
            return None
        unsigned = re.fullmatch(r"\(\s*unsigned\s+(.+?)\s+(==|!=|<=|>=|<|>)\s+(.+)\)", value)
        if unsigned:
            left, relation, right = unsigned.groups()
            compare_type = self.word_type
            if flag_source is not None:
                compare_ops = self._split_operands(flag_source.operands)
                if compare_ops:
                    operand = compare_ops[0].strip().lower()
                    size = re.match(r"(byte|word|dword|qword)\s+ptr\b", operand)
                    if size:
                        compare_type = _SIZE_TYPE[size.group(1)]
                    else:
                        token = operand.split()[-1]
                        if token in {"al", "ah", "bl", "bh", "cl", "ch", "dl", "dh"} or re.fullmatch(r"r(?:8|9|1[0-5])b", token):
                            compare_type = "uint8_t"
                        elif token in {"ax", "bx", "cx", "dx", "si", "di", "bp", "sp"} or re.fullmatch(r"r(?:8|9|1[0-5])w", token):
                            compare_type = "uint16_t"
                        elif token in {"eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp"} or re.fullmatch(r"r(?:8|9|1[0-5])d", token):
                            compare_type = "uint32_t"
            left = f"(({compare_type})({left}))"
            right = f"(({compare_type})({right}))"
            value = f"({left} {relation} {right})"

        def replace_register(match: re.Match[str]) -> str:
            return self._operand(match.group(1), ins, argument_families, constants)

        value = _REGISTER_RE.sub(replace_register, value)
        # Reject data-flow-only atoms that are not C expressions instead of
        # leaking internal names into the pseudocode.
        if re.search(r"\b(?:unknown|opaque|static_ptr|mem)\b", value):
            return None
        return value

    def _update_constants(self, ins: Instruction, constants: dict[str, int]) -> None:
        m = ins.mnemonic.lower()
        ops = self._split_operands(ins.operands)
        if not ops:
            return
        first = ops[0].strip().lower()
        family = _REGISTER_FAMILIES.get(first)
        if not family or m in {"cmp", "test", "push", "call", "jmp", "ret", "nop", "leave"} or m.startswith("j"):
            return
        # A write through an alias invalidates/updates the same variable family.
        if m in {"mov", "movabs", "movzx", "movsx", "movsxd"} and len(ops) >= 2:
            immediate = _parse_int(ops[1])
            if immediate is not None:
                constants[family] = immediate
            else:
                source_family = _REGISTER_FAMILIES.get(ops[1].strip().lower())
                if source_family in constants:
                    constants[family] = constants[source_family]
                else:
                    constants.pop(family, None)
            return
        if m == "xor" and len(ops) >= 2 and ops[0].strip().lower() == ops[1].strip().lower():
            constants[family] = 0
            return
        if m in _BINARY and len(ops) >= 2:
            rhs = _parse_int(ops[-1])
            lhs = constants.get(family)
            if lhs is None:
                constants.pop(family, None)
            elif rhs is not None:
                folded = self._const_eval(m, lhs, rhs)
                if folded is None: constants.pop(family, None)
                else: constants[family] = folded
            else:
                source_family = _REGISTER_FAMILIES.get(ops[-1].strip().lower())
                if source_family in constants:
                    folded = self._const_eval(m, lhs, constants[source_family])
                    if folded is not None:
                        constants[family] = folded
                    else:
                        constants.pop(family, None)
                else:
                    constants.pop(family, None)
            return
        if m in {"inc", "dec"}:
            old = constants.get(family)
            if old is not None:
                constants[family] = old + (1 if m == "inc" else -1)
            return
        constants.pop(family, None)

    def lift(self, ins: Instruction, known_labels: set[int], condition_expression: str | None = None,
             argument_families: dict[str, str] | None = None,
             constants: dict[str, int] | None = None,
             call_arguments: list[str] | None = None,
             call_returns_value: bool = False,
             function_return_type: str | None = None) -> str | None:
        m = ins.mnemonic.lower()
        raw = self._split_operands(ins.operands)
        argument_families = argument_families or {}
        constants = constants or {}
        if m in {"nop", "endbr64", "endbr32", ".byte"}:
            return None
        if m in {"push", "pop", "leave"}:
            if m == "pop" and raw:
                dst = self._operand(raw[0], ins, argument_families, constants, destination=True)
                return f"{dst} = stack_pop(); /* reconstructed stack operation */"
            return f"/* machine-level stack operation: {m} {ins.operands} */"
        if m in {"ret", "retq", "retn", "retf", "iret", "iretq"}:
            if function_return_type not in {None, "void"}:
                value = self._active_register_values.get("rax") if self._active_register_values is not None else None
                return f"return {value};" if _stable_inline_value(value, self._immutable_arguments) else "return v_rax;"
            if function_return_type == "void":
                return "return;"
            return "return /* return-register/type recovery pending */;"
        if m in {"mov", "movabs", "movzx", "movsx", "movsxd"} and len(raw) == 2:
            dst = self._operand(raw[0], ins, argument_families, constants, destination=True)
            src = self._operand(raw[1], ins, argument_families, constants)
            if m in {"movzx", "movsx", "movsxd"}:
                dst_width = "uint64_t" if self.bits == 64 else "uint32_t"
                src_width = next((_SIZE_TYPE[p] for p in ("byte", "word", "dword", "qword") if p in raw[1].lower()), None)
                if src_width:
                    src = f"({dst_width})({src})"
            return f"{dst} = {src};"
        if m == "lea" and len(raw) == 2:
            dst = self._operand(raw[0], ins, argument_families, constants, destination=True)
            if ins.reference_target in self.strings:
                src = json.dumps(self.strings[ins.reference_target], ensure_ascii=False)
            elif ins.reference_target is not None:
                src = f"(uintptr_t)0x{ins.reference_target:X}"
            else:
                src = f"address_of({self._operand(raw[1], ins, argument_families, constants)})"
            return f"{dst} = {src};"
        if m in {"sub", "add"} and raw and raw[0].strip().lower() in {"rsp", "esp"}:
            return f"/* stack pointer adjustment: {ins.text} */"
        if m in _BINARY and len(raw) >= 2:
            dst = self._operand(raw[0], ins, argument_families, constants, destination=True)
            if len(raw) == 2:
                lhs = self._operand(raw[0], ins, argument_families, constants)
                rhs = self._operand(raw[1], ins, argument_families, constants)
            else:
                lhs = self._operand(raw[1], ins, argument_families, constants)
                rhs = self._operand(raw[2], ins, argument_families, constants)
            if m == "xor" and raw[0].strip().lower() == raw[1].strip().lower():
                return f"{dst} = 0;"
            left_value = _parse_int(lhs)
            right_value = _parse_int(rhs)
            if left_value is not None and right_value is not None:
                folded = self._const_eval(m, left_value, right_value)
                if folded is not None:
                    return f"{dst} = {folded}; /* constant-folded */"
            return f"{dst} = {lhs} {_BINARY[m]} {rhs};"
        if m in {"inc", "dec", "neg", "not"} and raw:
            dst = self._operand(raw[0], ins, argument_families, constants, destination=True)
            src = self._operand(raw[0], ins, argument_families, constants)
            if m == "inc": return f"{dst} = {src} + 1;"
            if m == "dec": return f"{dst} = {src} - 1;"
            if m == "neg": return f"{dst} = -({src});"
            return f"{dst} = ~({src});"
        if m in {"cmp", "test", "ucomiss", "comiss", "comisd", "ucomisd"}:
            args = ", ".join(self._operand(x, ins, argument_families, constants) for x in raw)
            return f"/* condition flags set by {m}({args}); consumed by a later branch */"
        if m in {"call", "callq", "lcall"}:
            target = ins.direct_target
            args_text = ", ".join(call_arguments or [])
            if ins.import_name:
                full_name = ins.import_name
                name = full_name.split("!")[-1]
                call = f"{name}({args_text})"
                if call_returns_value:
                    return f"v_rax = {call}; /* import: {full_name} */"
                return f"{call}; /* import: {full_name} */"
            if target is None and ins.reference_target in self.imports:
                full_name = self.imports[ins.reference_target]
                name = full_name.split("!")[-1]
                call = f"{name}({args_text})"
                if call_returns_value:
                    return f"v_rax = {call}; /* import via IAT: {full_name} */"
                return f"{call}; /* import via IAT: {full_name} */"
            if target is not None:
                name = self.function_names.get(target, f"sub_{target:X}")
                call = f"{name}({args_text})"
                if call_returns_value:
                    return f"v_rax = {call};"
                return f"{call};"
            call = f"call_indirect({raw[0] if raw else 'unknown'}{', ' if args_text else ''}{args_text})"
            return f"v_rax = {call}; /* destination unresolved */" if call_returns_value else f"{call}; /* destination unresolved */"
        if ins.flow == "jump":
            if ins.direct_target is not None:
                label = f"loc_{ins.direct_target:X}" if ins.direct_target in known_labels else f"address_{ins.direct_target:X}"
                return f"goto {label};"
            if ins.import_name or ins.reference_target in self.imports:
                name = ins.import_name or self.imports.get(ins.reference_target, "unknown import")
                return f"/* tail-call/import jump: {name} */"
            if ins.resolved_indirect_jump_target is not None:
                target = ins.resolved_indirect_jump_target
                label = f"loc_{target:X}" if target in known_labels else f"address_{target:X}"
                return f"goto {label}; /* resolved indirect tail-jump target */"
            if (ins.jump_table_case_count is not None and ins.jump_table_case_count > 0
                    and ins.jump_table_index_register and ins.jump_table_slots is not None
                    and len(ins.jump_table_slots) >= ins.jump_table_case_count):
                index = self._operand(ins.jump_table_index_register, ins, argument_families, constants)
                bias = ins.jump_table_index_bias or 0
                selector = index if bias == 0 else f"({index} {'+' if bias > 0 else '-'} {abs(bias)})"
                cases: list[str] = []
                for slot_index, target in enumerate(ins.jump_table_slots[:ins.jump_table_case_count]):
                    case_value = slot_index + bias
                    if target is None:
                        destination = f"goto_indirect({raw[0] if raw else 'unknown'}); /* unresolved table slot */"
                    else:
                        label = f"loc_{target:X}" if target in known_labels else f"address_{target:X}"
                        destination = f"goto {label};"
                    cases.append(f"case {case_value}: {destination}")
                default_target = ins.jump_table_default_target
                if default_target is None:
                    default = f"goto_indirect({raw[0] if raw else 'unknown'});"
                else:
                    label = f"loc_{default_target:X}" if default_target in known_labels else f"address_{default_target:X}"
                    default = f"goto {label};"
                cases.append(f"default: {default}")
                return f"switch ({selector}) {{ {' '.join(cases)} }}"
            return f"goto_indirect({raw[0] if raw else 'unknown'});"
        if ins.flow == "conditional":
            target = ins.direct_target
            cond = _CONDITION.get(m, m.replace("j", ""))
            condition = condition_expression or f"condition_{cond}"
            if target is not None:
                label = f"loc_{target:X}" if target in known_labels else f"address_{target:X}"
                return f"if ({condition}) goto {label};"
            return f"if ({condition}) goto_indirect({raw[0] if raw else 'unknown'});"
        if m.startswith("cmov") and len(raw) >= 2:
            dst = self._operand(raw[0], ins, argument_families, constants, destination=True)
            src = self._operand(raw[1], ins, argument_families, constants)
            cond = m[4:]
            return f"if (condition_{cond}) {dst} = {src};"
        if m.startswith("set") and raw:
            dst = self._operand(raw[0], ins, argument_families, constants, destination=True)
            return f"{dst} = (uint8_t)(condition_{m[3:]});"
        return f"/* unmodeled instruction: {ins.text} */"

    def _x64_stack_entry_states(self, entry: int, blocks: list[BasicBlock]) -> dict[int, tuple[int | None, dict[int, str], dict[str, str | None]]]:
        """Recover must-known outgoing stack slots and RSP depth across CFG edges.

        Immediate qword stores and pushes, plus values copied through simple
        register operations, flow across blocks. At joins, a slot, register fact,
        or stack depth survives only when every reachable path agrees.
        """
        if self.bits != 64 or entry not in {block.start for block in blocks}:
            return {}
        block_by_start = {block.start: block for block in blocks}
        arg_families = self._argument_register_families(blocks)
        incoming: dict[int, tuple[int | None, dict[int, str], dict[str, str | None]]] = {
            entry: (0, {}, dict(arg_families))
        }
        outgoing: dict[int, tuple[int | None, dict[int, str], dict[str, str | None]]] = {}
        worklist = deque([entry])
        queued = {entry}

        def merge(left: tuple[int | None, dict[int, str], dict[str, str | None]],
                  right: tuple[int | None, dict[int, str], dict[str, str | None]]) -> tuple[int | None, dict[int, str], dict[str, str | None]]:
            left_depth, left_slots, left_registers = left
            right_depth, right_slots, right_registers = right
            depth = left_depth if left_depth == right_depth else None
            slots = {key: value for key, value in left_slots.items()
                     if right_slots.get(key) == value}
            registers = {
                key: (left_registers.get(key) if left_registers.get(key) == right_registers.get(key) else None)
                for key in left_registers.keys() | right_registers.keys()
            }
            return depth, slots, registers

        def resolve_stable_value(operand: str, registers: dict[str, str | None]) -> str | None:
            immediate = _parse_int(operand)
            if immediate is not None:
                return str(immediate)
            family = _REGISTER_FAMILIES.get(operand.strip().lower())
            return registers.get(family) if family else None

        def update_register_fact(ins: Instruction, registers: dict[str, str | None]) -> None:
            written = self._written_register_family(ins)
            if not written:
                return
            family, alias = written
            raw = self._split_operands(ins.operands)
            mnemonic = ins.mnemonic.lower()
            if mnemonic == "xor" and len(raw) >= 2 and raw[0].strip().lower() == raw[1].strip().lower():
                registers[family] = "0"
                return
            if mnemonic in {"mov", "movabs", "movzx", "movsx", "movsxd"} and len(raw) >= 2:
                value = resolve_stable_value(raw[1], registers)
                if value is not None:
                    number = _parse_int(value)
                    if alias in {"eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp"} or re.fullmatch(r"r(?:[89]|1[0-5])d", alias):
                        if number is not None:
                            value = str(number & 0xFFFFFFFF)
                        else:
                            value = f"((uint32_t)({value}))"
                    registers[family] = value
                else:
                    registers[family] = None
                return
            if mnemonic in _BINARY and len(raw) >= 2:
                old = registers.get(family)
                right = resolve_stable_value(raw[-1], registers)
                if old is None or right is None:
                    registers[family] = None
                    return
                left_number, right_number = _parse_int(old), _parse_int(right)
                if left_number is not None and right_number is not None:
                    result = self._const_eval(mnemonic, left_number, right_number)
                    if result is None:
                        registers[family] = None
                    else:
                        if alias in {"eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp"} or re.fullmatch(r"r(?:[89]|1[0-5])d", alias):
                            result &= 0xFFFFFFFF
                        registers[family] = str(result)
                else:
                    registers[family] = f"({old} {_BINARY[mnemonic]} {right})"
                return
            registers[family] = None

        while worklist:
            start = worklist.popleft()
            queued.discard(start)
            block = block_by_start[start]
            depth = incoming[start][0]
            slots = dict(incoming[start][1])
            registers = dict(incoming[start][2])
            for ins in block.instructions:
                mnemonic = ins.mnemonic.lower()
                raw = self._split_operands(ins.operands)
                if ins.flow == "call" or mnemonic in {"call", "callq", "lcall"}:
                    slots.clear()
                    for family in self._CALLER_SAVED:
                        registers[family] = None
                    continue
                if mnemonic == "push" and raw:
                    if depth is None:
                        slots.clear()
                    else:
                        key = -(depth + 8)
                        value = resolve_stable_value(raw[0], registers)
                        for existing_key in list(slots):
                            if key < existing_key + 8 and existing_key < key + 8:
                                slots.pop(existing_key, None)
                        if value is None:
                            slots.pop(key, None)
                        else:
                            slots[key] = value
                    depth = depth + 8 if depth is not None else None
                    continue
                if mnemonic == "pop":
                    destination_family = _REGISTER_FAMILIES.get(raw[0].strip().lower()) if raw else None
                    if depth is None or (raw and raw[0].strip().lower() == "rsp"):
                        depth = None
                        slots.clear()
                    else:
                        slots.pop(-depth, None)
                        depth -= 8
                    if raw:
                        if destination_family:
                            registers[destination_family] = None
                        elif "[" in raw[0]:
                            slots.clear()
                    continue
                if mnemonic == "leave":
                    depth = None
                    slots.clear()
                    registers["rbp"] = None
                    registers["rsp"] = None
                    continue
                if mnemonic in {"sub", "add"} and len(raw) >= 2 and raw[0].strip().lower() == "rsp":
                    amount = _parse_int(raw[1])
                    if amount is None or depth is None:
                        depth = None
                        slots.clear()
                    else:
                        depth += amount if mnemonic == "sub" else -amount
                    continue
                if mnemonic == "mov" and len(raw) >= 2 and "[" in raw[0]:
                    slot = re.fullmatch(
                        r"\s*qword\s+ptr\s+\[\s*rsp\s*([+-])\s*(0x[0-9a-f]+|\d+)\s*\]\s*",
                        raw[0], re.I,
                    )
                    value = resolve_stable_value(raw[1], registers)
                    if slot and depth is not None:
                        displacement = int(slot.group(2), 0)
                        if slot.group(1) == "-":
                            displacement = -displacement
                        key = displacement - depth
                        for existing_key in list(slots):
                            if key < existing_key + 8 and existing_key < key + 8:
                                slots.pop(existing_key, None)
                        if value is not None:
                            slots[key] = value
                    else:
                        slots.clear()
                    continue
                if raw and raw[0].strip().lower() == "rsp":
                    depth = None
                    slots.clear()
                    continue
                if raw and "[" in raw[0]:
                    slots.clear()
                if mnemonic == "xchg":
                    for operand in raw[:2]:
                        family = _REGISTER_FAMILIES.get(operand.strip().lower())
                        if family:
                            registers[family] = None
                else:
                    update_register_fact(ins, registers)

            result = (depth, slots, registers)
            if outgoing.get(start) == result:
                continue
            outgoing[start] = result
            for successor in block.successors:
                if successor not in block_by_start:
                    continue
                previous = incoming.get(successor)
                candidate = result if previous is None else merge(previous, result)
                if previous != candidate:
                    incoming[successor] = candidate
                    if successor not in queued:
                        worklist.append(successor)
                        queued.add(successor)
        return incoming

    def _x86_push_entry_states(self, entry: int, blocks: list[BasicBlock],
                               argument_families: dict[str, str]) -> dict[int, tuple[list[str | None], dict[str, str | None], int | None, dict[int, str]]]:
        """Propagate x86 pushes, ESP-relative outgoing slots, and register facts."""
        if self.bits != 32 or entry not in {block.start for block in blocks}:
            return {}
        block_by_start = {block.start: block for block in blocks}
        initial_registers: dict[str, str | None] = dict(argument_families)
        incoming: dict[int, tuple[list[str | None], dict[str, str | None], int | None, dict[int, str]]] = {
            entry: ([], initial_registers, 0, {})
        }
        outgoing: dict[int, tuple[list[str | None], dict[str, str | None], int | None, dict[int, str]]] = {}
        worklist = deque([entry])
        queued = {entry}

        def merge(left: tuple[list[str | None], dict[str, str | None], int | None, dict[int, str]],
                  right: tuple[list[str | None], dict[str, str | None], int | None, dict[int, str]]) -> tuple[list[str | None], dict[str, str | None], int | None, dict[int, str]]:
            left_pending, left_registers, left_depth, left_slots = left
            right_pending, right_registers, right_depth, right_slots = right
            if len(left_pending) == len(right_pending):
                pending = [a if a == b else None for a, b in zip(left_pending, right_pending)]
            elif max(len(left_pending), len(right_pending)) < 16:
                # Push order is chronological: the oldest push is the last
                # source-level argument. Keep agreeing positions and mark
                # absent or conflicting pushes unknown instead of discarding
                # the whole call-site stack state.
                width = max(len(left_pending), len(right_pending))
                pending = [
                    left_pending[index]
                    if index < len(left_pending) and index < len(right_pending)
                    and left_pending[index] == right_pending[index]
                    else None
                    for index in range(width)
                ]
            else:
                pending = []
            registers = {
                family: (left_registers.get(family) if left_registers.get(family) == right_registers.get(family) else None)
                for family in left_registers.keys() | right_registers.keys()
            }
            depth = left_depth if left_depth == right_depth else None
            slots = {key: value for key, value in left_slots.items()
                     if right_slots.get(key) == value}
            if depth is None:
                slots = {}
            return pending, registers, depth, slots

        def stable_value(operand: str, registers: dict[str, str | None]) -> str | None:
            immediate = _parse_int(operand)
            if immediate is not None:
                return str(immediate)
            family = _REGISTER_FAMILIES.get(operand.strip().lower())
            return registers.get(family) if family else None

        while worklist:
            start = worklist.popleft()
            queued.discard(start)
            block = block_by_start[start]
            pending = list(incoming[start][0])
            registers = dict(incoming[start][1])
            depth = incoming[start][2]
            slots = dict(incoming[start][3])
            constants = {
                family: number for family, value in registers.items()
                if value is not None and (number := _parse_int(value)) is not None
            }
            for ins in block.instructions:
                mnemonic = ins.mnemonic.lower()
                raw = self._split_operands(ins.operands)
                if ins.flow == "call" or mnemonic in {"call", "callq", "lcall"}:
                    pending.clear()
                    slots.clear()
                    target = ins.direct_target
                    if target is None and len(ins.indirect_call_targets or []) == 1:
                        target = ins.indirect_call_targets[0]
                    target_signature = self.function_signatures.get(target) if target is not None else None
                    cleanup = target_signature.get("callee_pop_bytes") if target_signature else None
                    if isinstance(cleanup, int) and depth is not None:
                        depth -= cleanup
                    for family in ("rax", "rcx", "rdx"):
                        registers[family] = None
                        constants.pop(family, None)
                    continue
                if mnemonic == "push" and raw:
                    pending.append(stable_value(raw[0], registers))
                    if len(pending) > 16:
                        pending.pop(0)
                    if depth is None:
                        slots.clear()
                    else:
                        depth += 4
                        slots[-depth] = stable_value(raw[0], registers) or "?"
                    continue
                if mnemonic == "pop":
                    if pending:
                        pending.pop()
                    if depth is not None:
                        slots.pop(-depth, None)
                        depth -= 4
                    if raw:
                        family = _REGISTER_FAMILIES.get(raw[0].strip().lower())
                        if family:
                            registers[family] = None
                            constants.pop(family, None)
                    continue
                if mnemonic == "leave":
                    pending.clear()
                    depth = None
                    slots.clear()
                    registers["rbp"] = registers["rsp"] = None
                    constants.pop("rbp", None)
                    constants.pop("rsp", None)
                    continue
                if mnemonic in {"add", "lea"} and raw and raw[0].strip().lower() == "esp":
                    adjustment = _parse_int(raw[1]) if mnemonic == "add" and len(raw) > 1 else None
                    if mnemonic == "lea" and len(raw) > 1:
                        match = re.fullmatch(r"\[\s*esp\s*\+\s*(0x[0-9a-f]+|\d+)\s*\]", raw[1], re.I)
                        adjustment = int(match.group(1), 0) if match else None
                    if adjustment is not None and adjustment > 0 and adjustment % 4 == 0:
                        del pending[-min(adjustment // 4, len(pending)):]
                        if depth is not None:
                            depth -= adjustment
                            slots = {key: value for key, value in slots.items() if key >= -depth}
                    else:
                        pending.clear()
                        depth = None
                        slots.clear()
                    registers["rsp"] = None
                    constants.pop("rsp", None)
                    continue

                if mnemonic in {"sub", "add"} and len(raw) >= 2 and raw[0].strip().lower() == "esp":
                    amount = _parse_int(raw[1])
                    if amount is None or amount % 4 or depth is None:
                        depth = None
                        slots.clear()
                    elif mnemonic == "sub":
                        depth += amount
                    else:
                        depth -= amount
                        slots = {key: value for key, value in slots.items() if key >= -depth}
                    registers["rsp"] = None
                    constants.pop("rsp", None)
                    continue

                if mnemonic == "mov" and len(raw) >= 2 and "[" in raw[0]:
                    match = re.fullmatch(
                        r"\s*(?:(?:dword)\s+ptr\s+)?\[\s*esp\s*([+-])?\s*(0x[0-9a-f]+|\d+)?\s*\]\s*",
                        raw[0], re.I,
                    )
                    if match and depth is not None:
                        sign, displacement_text = match.groups()
                        displacement = _parse_int(displacement_text) if displacement_text else 0
                        if displacement is None:
                            slots.clear()
                        else:
                            key = (-displacement if sign == "-" else displacement) - depth
                            value = stable_value(raw[1], registers)
                            for existing_key in list(slots):
                                if key < existing_key + 4 and existing_key < key + 4:
                                    slots.pop(existing_key, None)
                            if value is not None:
                                slots[key] = value
                    else:
                        slots.clear()
                    continue

                written = self._written_register_family(ins)
                if not written:
                    continue
                family, alias = written
                value: str | None = None
                if mnemonic in {"mov", "movabs", "movzx", "movsx", "movsxd"} and len(raw) >= 2:
                    value = stable_value(raw[1], registers)
                elif mnemonic == "lea" and ins.reference_target is not None:
                    value = str(ins.reference_target)
                elif mnemonic == "xor" and len(raw) >= 2 and raw[0].strip().lower() == raw[1].strip().lower():
                    value = "0"
                elif mnemonic in _BINARY and len(raw) >= 2:
                    left = _parse_int(registers.get(family) or "")
                    right = _parse_int(stable_value(raw[-1], registers) or "")
                    if left is not None and right is not None:
                        result = self._const_eval(mnemonic, left, right)
                        if result is not None:
                            value = str(result & 0xFFFFFFFF)
                registers[family] = value
                if value is not None and (number := _parse_int(value)) is not None:
                    constants[family] = number
                else:
                    constants.pop(family, None)

            result = (pending, registers, depth, slots)
            if outgoing.get(start) == result:
                continue
            outgoing[start] = result
            for successor in block.successors:
                if successor not in block_by_start:
                    continue
                previous = incoming.get(successor)
                candidate = result if previous is None else merge(previous, result)
                if previous != candidate:
                    incoming[successor] = candidate
                    if successor not in queued:
                        worklist.append(successor)
                        queued.add(successor)
        return incoming

    def function_text(self, name: str, entry: int, blocks: list[BasicBlock],
                      dataflow_conditions: dict[int, str] | None = None) -> str:
        if not blocks:
            return f"/* No decoded blocks for {name} at 0x{entry:X}. */\n"
        self._local_types = {}
        self._active_register_values = None
        self._immutable_arguments = set()
        labels = {b.start for b in blocks}
        self._operand_stack_depths = {}
        self._has_stack_depth_context = self.bits == 64
        if self.bits == 64:
            ordered_blocks = sorted(blocks, key=lambda block: block.start)
            entry_block = next((block for block in ordered_blocks if block.start == entry), None)
            rbp_depth: int | None = None
            if entry_block is not None:
                depth = 0
                linear = True
                for ins in entry_block.instructions:
                    mnemonic = ins.mnemonic.lower()
                    operands = self._split_operands(ins.operands)
                    if linear:
                        bases = {"rsp": depth}
                        if rbp_depth is not None:
                            bases["rbp"] = rbp_depth
                        self._operand_stack_depths[ins.address] = bases
                    if mnemonic in {"call", "jmp", "ret", "retn"} or mnemonic.startswith("j"):
                        linear = False
                        continue
                    if len(operands) >= 2 and mnemonic == "mov" and operands[0].strip().lower() == "rbp" and operands[1].strip().lower() == "rsp":
                        rbp_depth = depth
                    if mnemonic == "push":
                        depth += 8
                    elif mnemonic == "pop":
                        depth -= 8
                    elif mnemonic in {"sub", "add"} and len(operands) >= 2 and operands[0].strip().lower() == "rsp":
                        amount = _parse_int(operands[1])
                        if amount is None:
                            linear = False
                        else:
                            depth += amount if mnemonic == "sub" else -amount
                    elif operands and operands[0].strip().lower() == "rsp":
                        linear = False
                if rbp_depth is not None:
                    for block in ordered_blocks:
                        if block is entry_block:
                            continue
                        for ins in block.instructions:
                            self._operand_stack_depths[ins.address] = {"rbp": rbp_depth}
        arg_families = self._argument_register_families(blocks)
        written_families = {
            family
            for block in blocks
            for ins in block.instructions
            if (written := self._written_register_family(ins)) is not None
            for family in [written[0]]
        }
        self._immutable_arguments = {
            argument for family, argument in arg_families.items()
            if family not in written_families
        }
        x64_stack_entry_states = self._x64_stack_entry_states(entry, blocks)
        x86_push_entry_states = self._x86_push_entry_states(entry, blocks, arg_families)
        signature = self.function_signatures.get(entry, {})
        arg_count = int(signature.get("register_parameter_count", max((int(n[3:]) for n in arg_families.values()), default=0)))
        stack_parameters = signature.get("stack_parameters", [])
        register_parameters = signature.get("register_parameters", [])
        if register_parameters:
            arg_declarations = [
                f"{row.get('type', self.word_type)} {row['name']}" for row in register_parameters
                if isinstance(row, dict) and row.get("name")
            ]
        else:
            arg_declarations = [f"{self.word_type} arg{i}" for i in range(1, arg_count + 1)]
        arg_declarations.extend(f"{self.word_type} {name}" for name in stack_parameters)
        args = ", ".join(arg_declarations) or "void"
        return_type = str(signature.get("return_type", "void"))
        body: list[str] = []
        for block in sorted(blocks, key=lambda b: b.start):
            if block.start != entry or len(blocks) > 1:
                body.append(f"loc_{block.start:X}:")
            # General expression propagation remains block-local. Outgoing stack
            # slots use a separate must-analysis across CFG edges.
            incoming_stack_state = x64_stack_entry_states.get(block.start, (0, {}, {}))
            x86_push_state = x86_push_entry_states.get(block.start, ([], {}, 0, {}))
            incoming_register_values = {
                family: value for family, value in (
                    incoming_stack_state[2] if self.bits == 64 else x86_push_state[1]
                ).items() if value is not None
            }
            constants: dict[str, int] = {
                family: number
                for family, value in incoming_register_values.items()
                if (number := _parse_int(value)) is not None
            }
            register_values: dict[str, str] = incoming_register_values
            self._active_register_values = register_values
            written_since_call: set[str] = {
                family for family, argument in arg_families.items()
                if incoming_register_values.get(family) != argument
            }
            pending_stack_arguments: list[str | None] = (
                list(x86_push_state[0]) if self.bits == 32 else []
            )
            stack_depth: int | None = incoming_stack_state[0]
            outgoing_stack_values: dict[int, str] = (
                dict(incoming_stack_state[1]) if self.bits == 64 and stack_depth is not None
                else dict(x86_push_state[3]) if self.bits == 32 and x86_push_state[2] is not None
                else {}
            )
            if self.bits == 32:
                stack_depth = x86_push_state[2]
            flag_source = None

            def outgoing_stack_value(source: str, ins: Instruction) -> str:
                family = _REGISTER_FAMILIES.get(source.strip().lower())
                if family and family not in register_values and family in written_since_call:
                    return f"v_{family}"
                return self._value_for_call_operand(
                    source, ins, arg_families, constants, register_values
                )

            for index, ins in enumerate(block.instructions):
                raw = self._split_operands(ins.operands)
                condition = self._condition_expression(ins, flag_source, arg_families, constants)
                flow_condition = (dataflow_conditions or {}).get(ins.address)
                if isinstance(flow_condition, str):
                    reconstructed_condition = self._format_dataflow_condition(
                        flow_condition, ins, arg_families, constants, flag_source
                    )
                    if reconstructed_condition is not None:
                        condition = reconstructed_condition
                is_call = ins.mnemonic.lower() in {"call", "callq", "lcall"}
                call_args: list[str] = []
                call_returns_value = False
                call_ins = ins
                if is_call:
                    if ins.direct_target is None and ins.import_name is None and raw:
                        target_family = _REGISTER_FAMILIES.get(raw[0].strip().lower())
                        target_value = register_values.get(target_family, "") if target_family else ""
                        import_match = re.fullmatch(r"__iat_import__:(.+)", target_value)
                        if import_match:
                            ins.import_name = import_match.group(1)
                    if ins.import_name is None and ins.direct_target is None and len(ins.indirect_call_targets or []) == 1:
                        call_ins = replace(ins, direct_target=ins.indirect_call_targets[0])
                    call_args = self._call_arguments(
                        call_ins, arg_families, constants, register_values, written_since_call,
                        pending_stack_arguments, outgoing_stack_values,
                        stack_depth if stack_depth is not None else 0,
                    )
                    # Assign the call's value only when it is used before RAX is
                    # overwritten, or the current function forwards it directly to ret.
                    call_returns_value = self._is_rax_read_after_call(
                        block, index, return_type not in {"void", "none"}
                    )
                stmt = self.lift(call_ins, labels, condition, arg_families, constants,
                                 call_arguments=call_args, call_returns_value=call_returns_value,
                                 function_return_type=return_type)
                # Pure register copies and address calculations are represented
                # by the symbolic value state below. Omitting the dead machine
                # temporary assignment gives later expressions a source-like
                # form without suppressing memory reads or changing flags.
                written = self._written_register_family(ins)
                if stmt and written and written[0] not in arg_families and written[0] not in {"rsp", "rbp"}:
                    mnemonic = ins.mnemonic.lower()
                    source_family = _REGISTER_FAMILIES.get(raw[1].strip().lower()) if len(raw) >= 2 else None
                    source_is_constant = (
                        len(raw) >= 2 and (
                            _parse_int(raw[1]) is not None
                            or source_family in constants
                            or _stable_inline_value(
                                register_values.get(source_family) if source_family else None,
                                self._immutable_arguments,
                            )
                        )
                    )
                    static_address = mnemonic == "lea" and ins.reference_target is not None
                    if (mnemonic in {"mov", "movabs", "movzx", "movsx", "movsxd"}
                            and source_is_constant and "[" not in raw[1]
                            or static_address):
                        stmt = None
                if stmt:
                    body.append("    " + stmt)
                self._update_constants(ins, constants)
                if is_call:
                    # The Windows x64 caller-saved register set may be clobbered by a call.
                    for family in self._CALLER_SAVED:
                        constants.pop(family, None)
                        register_values.pop(family, None)
                    if call_returns_value:
                        register_values["rax"] = "v_rax"
                    written_since_call.clear()
                    if self.bits == 64:
                        outgoing_stack_values.clear()
                    if self.bits == 32:
                        outgoing_stack_values.clear()
                        target_signature = self.function_signatures.get(ins.direct_target) if ins.direct_target is not None else None
                        if target_signature is not None:
                            consumed = len(target_signature.get("stack_parameters", []))
                        else:
                            consumed = self._IMPORT_ARITY.get(self._base_symbol(self._import_name_for(ins)), 0)
                        if consumed:
                            del pending_stack_arguments[-consumed:]
                        cleanup = target_signature.get("callee_pop_bytes") if target_signature else None
                        if isinstance(cleanup, int) and stack_depth is not None:
                            stack_depth -= cleanup
                else:
                    if self.bits == 32 and ins.mnemonic.lower() == "push" and raw:
                        pending_stack_arguments.append(
                            self._operand(raw[0], ins, arg_families, constants)
                        )
                    elif self.bits == 32 and ins.mnemonic.lower() == "pop":
                        if pending_stack_arguments:
                            pending_stack_arguments.pop()
                    elif self.bits == 32 and ins.mnemonic.lower() in {"add", "lea"} and raw:
                        stack_adjustment = None
                        if ins.mnemonic.lower() == "add" and len(raw) >= 2 and raw[0].strip().lower() == "esp":
                            stack_adjustment = _parse_int(raw[1])
                        elif ins.mnemonic.lower() == "lea" and len(raw) >= 2 and raw[0].strip().lower() == "esp":
                            match = re.fullmatch(r"\[\s*esp\s*\+\s*(0x[0-9a-f]+|\d+)\s*\]", raw[1].strip(), re.I)
                            stack_adjustment = int(match.group(1), 0) if match else None
                        if stack_adjustment is not None and stack_adjustment > 0 and stack_adjustment % 4 == 0:
                            del pending_stack_arguments[-min(stack_adjustment // 4, len(pending_stack_arguments)):]
                        elif stack_adjustment is not None:
                            pending_stack_arguments.clear()
                    elif self.bits == 32 and ins.mnemonic.lower() == "mov" and raw and raw[0].strip().lower() == "esp":
                        pending_stack_arguments.clear()
                    if self.bits == 32:
                        mnemonic = ins.mnemonic.lower()
                        if mnemonic == "push" and raw:
                            if stack_depth is None:
                                outgoing_stack_values.clear()
                            else:
                                slot_key = -(stack_depth + 4)
                                outgoing_stack_values[slot_key] = outgoing_stack_value(raw[0], ins)
                                stack_depth += 4
                        elif mnemonic == "pop":
                            if stack_depth is None:
                                outgoing_stack_values.clear()
                            else:
                                outgoing_stack_values.pop(-stack_depth, None)
                                stack_depth -= 4
                        elif mnemonic == "mov" and len(raw) >= 2 and "[" in raw[0]:
                            slot = re.fullmatch(
                                r"\s*(?:(?:dword)\s+ptr\s+)?\[\s*esp\s*([+-])?\s*(0x[0-9a-f]+|\d+)?\s*\]\s*",
                                raw[0], re.I,
                            )
                            if slot and stack_depth is not None:
                                sign, displacement_text = slot.groups()
                                displacement = _parse_int(displacement_text) if displacement_text else 0
                                if displacement is None:
                                    outgoing_stack_values.clear()
                                else:
                                    slot_key = (-displacement if sign == "-" else displacement) - stack_depth
                                    for existing_key in list(outgoing_stack_values):
                                        if slot_key < existing_key + 4 and existing_key < slot_key + 4:
                                            outgoing_stack_values.pop(existing_key, None)
                                    outgoing_stack_values[slot_key] = outgoing_stack_value(raw[1], ins)
                            else:
                                outgoing_stack_values.clear()
                        elif raw and "[" in raw[0]:
                            outgoing_stack_values.clear()

                        if mnemonic in {"sub", "add"} and len(raw) >= 2 and raw[0].strip().lower() == "esp":
                            amount = _parse_int(raw[1])
                            if amount is None or amount % 4 or stack_depth is None:
                                stack_depth = None
                                outgoing_stack_values.clear()
                            elif mnemonic == "sub":
                                stack_depth += amount
                            else:
                                stack_depth -= amount
                                outgoing_stack_values = {
                                    key: value for key, value in outgoing_stack_values.items()
                                    if key >= -stack_depth
                                }
                        elif mnemonic == "lea" and len(raw) >= 2 and raw[0].strip().lower() == "esp":
                            adjust = re.fullmatch(r"\[\s*esp\s*\+\s*(0x[0-9a-f]+|\d+)\s*\]", raw[1], re.I)
                            amount = _parse_int(adjust.group(1)) if adjust else None
                            if amount is None or stack_depth is None:
                                stack_depth = None
                                outgoing_stack_values.clear()
                            else:
                                stack_depth -= amount
                                outgoing_stack_values = {
                                    key: value for key, value in outgoing_stack_values.items()
                                    if key >= -stack_depth
                                }
                        elif mnemonic in {"leave"} or (raw and raw[0].strip().lower() == "esp"
                                                       and mnemonic not in {"cmp", "test"}):
                            stack_depth = None
                            outgoing_stack_values.clear()
                    if self.bits == 64:
                        mnemonic = ins.mnemonic.lower()
                        if mnemonic == "mov" and len(raw) >= 2:
                            destination = raw[0].strip()
                            slot = re.search(
                                r"\[\s*rsp\s*([+-])\s*(0x[0-9a-f]+|\d+)\s*\]",
                                destination, re.I,
                            )
                            if slot:
                                displacement = int(slot.group(2), 0)
                                if slot.group(1) == "-":
                                    displacement = -displacement
                                if stack_depth is None:
                                    outgoing_stack_values.clear()
                                else:
                                    slot_key = displacement - stack_depth
                                    slot_width = next((width for size, width in (
                                        ("byte", 1), ("word", 2), ("dword", 4), ("qword", 8)
                                    ) if re.match(rf"\s*{size}\s+ptr\b", destination, re.I)), 8)
                                    for existing_key in list(outgoing_stack_values):
                                        if (slot_key < existing_key + slot_width
                                                and existing_key < slot_key + slot_width):
                                            outgoing_stack_values.pop(existing_key, None)
                                    outgoing_stack_values[slot_key] = outgoing_stack_value(raw[1], ins)
                            elif "[" in destination:
                                outgoing_stack_values.clear()
                        elif mnemonic == "push" and raw:
                            # A push stores at entry-RSP minus the new depth.
                            # Keeping this coordinate lets the usual callee-slot
                            # mapping handle pushes followed by shadow-space setup.
                            if stack_depth is None:
                                outgoing_stack_values.clear()
                            else:
                                slot_key = -(stack_depth + 8)
                                for existing_key in list(outgoing_stack_values):
                                    if slot_key < existing_key + 8 and existing_key < slot_key + 8:
                                        outgoing_stack_values.pop(existing_key, None)
                                outgoing_stack_values[slot_key] = outgoing_stack_value(raw[0], ins)
                        elif mnemonic == "pop":
                            if stack_depth is None:
                                outgoing_stack_values.clear()
                            else:
                                outgoing_stack_values.pop(-stack_depth, None)
                        elif raw and "[" in raw[0]:
                            outgoing_stack_values.clear()
                        if mnemonic == "push":
                            stack_depth = stack_depth + 8 if stack_depth is not None else None
                        elif mnemonic == "pop":
                            stack_depth = stack_depth - 8 if stack_depth is not None else None
                        elif mnemonic == "leave":
                            stack_depth = None
                            outgoing_stack_values.clear()
                        elif mnemonic in {"sub", "add"} and len(raw) >= 2 and raw[0].strip().lower() == "rsp":
                            amount = _parse_int(raw[1])
                            if amount is not None and stack_depth is not None:
                                stack_depth += amount if mnemonic == "sub" else -amount
                            else:
                                stack_depth = None
                                outgoing_stack_values.clear()
                        elif raw and raw[0].strip().lower() == "rsp":
                            stack_depth = None
                            outgoing_stack_values.clear()
                    written = self._written_register_family(ins)
                    if written:
                        written_since_call.add(written[0])
                    self._update_register_values(ins, arg_families, constants, register_values)
                m = ins.mnemonic.lower()
                if m in {"cmp", "test"}:
                    flag_source = ins
                elif ins.flow == "conditional":
                    flag_source = None
                elif m in {"add", "sub", "and", "or", "xor", "imul", "mul", "inc", "dec", "neg", "shl", "shr", "sar", "sal", "sbb", "adc"}:
                    flag_source = None
                elif ins.flow == "call":
                    flag_source = None
        body = self._structure_control_flow(body, blocks)
        variable_pattern = re.compile(r"\b(v_(?:rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r(?:[0-9]|1[0-5]))|local_[0-9A-F]+|stack_arg_[0-9A-F]+)\b")
        found = set()
        for line in body:
            found.update(variable_pattern.findall(line))
        found.difference_update(str(name) for name in stack_parameters)
        declarations = [
            f"    {self._local_types.get(var, self.word_type)} {var};"
            for var in sorted(found)
        ]
        out = [
            "/* Independent decompiler output: heuristic C-like reconstruction. */",
            f"/* Function candidate {name} at 0x{entry:X}. Check control flow and types manually. */",
            "#include <stdint.h>",
            f"{return_type} {name}({args}) {{",
        ]
        if declarations:
            out.extend(declarations)
        out.extend(body)
        out += ["}", ""]
        result = "\n".join(out)
        result = PseudocodeLifter._prettify_c_syntax(result)
        self._local_types = {}
        self._active_register_values = None
        self._immutable_arguments = set()
        self._operand_stack_depths = {}
        self._has_stack_depth_context = False
        return result

    @staticmethod
    def _structure_control_flow(body: list[str], blocks: list[BasicBlock]) -> list[str]:
        """Apply only proven CFG transformations until no more progress is made."""
        for _ in range(max(1, len(blocks))):
            previous = tuple(body)
            body = PseudocodeLifter._structure_simple_switches(body, blocks)
            body = PseudocodeLifter._structure_simple_do_while(body, blocks)
            body = PseudocodeLifter._structure_simple_loops(body, blocks)
            body = PseudocodeLifter._structure_simple_diamonds(body, blocks)
            body = PseudocodeLifter._structure_single_if(body, blocks)
            if tuple(body) == previous:
                break
        body = PseudocodeLifter._clean_compiler_comments(body)
        body = PseudocodeLifter._remove_unreferenced_labels(body)
        return body

    @staticmethod
    def _structure_simple_switches(body: list[str], blocks: list[BasicBlock]) -> list[str]:
        """Convert a small equality-test dispatch chain into a switch statement."""
        block_by_start = {block.start: block for block in blocks}
        labels = {
            f"loc_{block.start:X}": index
            for index, line in enumerate(body)
            for block in blocks
            if line.strip() == f"loc_{block.start:X}:"
        }
        label_indices = sorted(labels.values())

        def block_lines(start: int) -> list[str]:
            index = labels[f"loc_{start:X}"]
            next_index = next((value for value in label_indices if value > index), len(body))
            return body[index + 1:next_index]

        ordered = sorted(blocks, key=lambda block: block.start)
        for dispatch in ordered:
            if not dispatch.instructions or dispatch.instructions[-1].flow != "conditional":
                continue
            if len(dispatch.successors) != 2:
                continue
            branch = dispatch.instructions[-1]
            case_start = branch.direct_target
            if case_start not in dispatch.successors:
                continue
            fallthrough_start = next(start for start in dispatch.successors if start != case_start)
            case_block = block_by_start.get(case_start)
            next_dispatch = block_by_start.get(fallthrough_start)
            if case_block is None or next_dispatch is None or case_block.successors == []:
                continue
            if len(case_block.successors) != 1:
                continue
            join_start = case_block.successors[0]
            cases = [(dispatch, case_block)]
            current = next_dispatch
            while (current.instructions and current.instructions[-1].flow == "conditional"
                   and len(current.successors) == 2):
                current_case_start = current.instructions[-1].direct_target
                if current_case_start not in current.successors:
                    break
                current_case = block_by_start.get(current_case_start)
                current_fallthrough = next(
                    start for start in current.successors if start != current_case_start
                )
                if current_case is None or len(current_case.successors) != 1:
                    break
                cases.append((current, current_case))
                current = block_by_start.get(current_fallthrough)
                if current is None:
                    break
            if len(cases) < 2 or current is None:
                continue
            default_block = current
            if default_block.start == join_start or len(default_block.successors) != 1:
                continue
            if join_start not in default_block.successors:
                join_start = default_block.successors[0]
            case_starts = {case_block.start for _dispatch, case_block in cases}
            valid_case_successors = case_starts | {join_start}
            if any(
                len(case_block.successors) != 1
                or case_block.successors[0] not in valid_case_successors
                for _dispatch, case_block in cases
            ):
                continue
            if default_block.successors[0] != join_start:
                continue
            all_labels = [f"loc_{block.start:X}" for pair in cases for block in pair]
            all_labels.extend([f"loc_{default_block.start:X}", f"loc_{join_start:X}"])
            if not all(label in labels for label in all_labels):
                continue
            start_index = labels[f"loc_{dispatch.start:X}"]
            join_index = labels[f"loc_{join_start:X}"]
            if start_index >= join_index:
                continue
            selector = dispatch.instructions[-2].operands.split(",", 1)[0].strip() if len(dispatch.instructions) >= 2 and dispatch.instructions[-2].mnemonic.lower() == "cmp" else None
            if not selector:
                continue
            selector = selector.replace("dword ptr ", "").replace("qword ptr ", "").strip()
            case_values: list[str] = []
            for case_dispatch, case_block in cases:
                cmp_instruction = next(
                    (ins for ins in reversed(case_dispatch.instructions) if ins.mnemonic.lower() == "cmp"),
                    None,
                )
                if cmp_instruction is None:
                    case_values = []
                    break
                operands = cmp_instruction.operands.split(",", 1)
                if len(operands) != 2:
                    case_values = []
                    break
                case_values.append(operands[1].strip())
            if len(case_values) != len(cases):
                continue
            structured = body[:start_index]
            structured.append(f"    switch ({selector}) {{")
            for value, (_dispatch, case_block) in zip(case_values, cases):
                next_case = case_block.successors[0]
                lines = [
                    line for line in block_lines(case_block.start)
                    if not line.strip().startswith(f"goto loc_{next_case:X};")
                ]
                structured.append(f"    case {value}:")
                structured.extend(f"        {line}" for line in lines)
                if next_case == join_start:
                    structured.append("        break;")
            default_lines = [
                line for line in block_lines(default_block.start)
                if not line.strip().startswith(f"goto loc_{join_start:X};")
            ]
            structured.append("    default:")
            structured.extend(f"        {line}" for line in default_lines)
            structured.append("        break;")
            structured.append("    }")
            structured.extend(body[join_index + 1:])
            return structured
        return body

    @staticmethod
    def _structure_simple_do_while(body: list[str], blocks: list[BasicBlock]) -> list[str]:
        """Convert a single-block post-tested loop into ``do { } while``."""
        block_by_start = {block.start: block for block in blocks}
        labels = {
            f"loc_{block.start:X}": index
            for index, line in enumerate(body)
            for block in blocks
            if line.strip() == f"loc_{block.start:X}:"
        }
        for loop_body in sorted(blocks, key=lambda block: block.start):
            if not loop_body.instructions or loop_body.instructions[-1].flow != "conditional":
                continue
            if len(loop_body.successors) != 2:
                continue
            back_target = loop_body.instructions[-1].direct_target
            if back_target != loop_body.start:
                continue
            exit_start = next(
                (start for start in loop_body.successors if start != loop_body.start),
                None,
            )
            if exit_start is None or exit_start not in block_by_start:
                continue
            body_label = f"loc_{loop_body.start:X}"
            exit_label = f"loc_{exit_start:X}"
            if body_label not in labels or exit_label not in labels:
                continue
            body_index = labels[body_label]
            exit_index = labels[exit_label]
            if body_index >= exit_index:
                continue
            loop_lines = body[body_index + 1:exit_index]
            branch_index = next(
                (
                    index for index in range(len(loop_lines) - 1, -1, -1)
                    if loop_lines[index].lstrip().startswith("if (")
                    and f"goto loc_{loop_body.start:X};" in loop_lines[index]
                ),
                None,
            )
            if branch_index is None:
                continue
            branch = loop_lines[branch_index].strip()
            match = re.fullmatch(
                rf"if \((.+)\) goto loc_{loop_body.start:X};",
                branch,
            )
            if match is None:
                continue
            structured = body[:body_index]
            structured.append("    do {")
            structured.extend(f"    {line}" for line in loop_lines[:branch_index])
            structured.append(f"    }} while ({match.group(1)});")
            structured.extend(body[exit_index + 1:])
            return structured
        return body

    @staticmethod
    def _structure_simple_loops(body: list[str], blocks: list[BasicBlock]) -> list[str]:
        """Structure an acyclic multi-block loop body with one header and exit.

        Internal labels/gotos are retained where needed, while explicit edges to
        the loop header and unique exit become ``continue`` and ``break``. Nested
        cycles and irreducible or noncontiguous regions remain in CFG form.
        """
        block_by_start = {block.start: block for block in blocks}
        labels = {
            f"loc_{block.start:X}": index
            for index, line in enumerate(body)
            for block in blocks
            if line.strip() == f"loc_{block.start:X}:"
        }
        for header in sorted(blocks, key=lambda block: block.start):
            if not header.instructions or header.instructions[-1].flow != "conditional":
                continue
            if len(header.successors) != 2:
                continue
            loop_body_start = header.instructions[-1].direct_target
            if loop_body_start not in header.successors:
                continue
            if loop_body_start not in block_by_start:
                continue
            exit_start = next(start for start in header.successors if start != loop_body_start)
            if exit_start in {header.start, loop_body_start} or exit_start not in block_by_start:
                continue

            # Collect the loop body, treating the header and unique exit as
            # boundaries. Every other reachable successor must be in the region.
            region: set[int] = set()
            worklist = [loop_body_start]
            valid_region = True
            while worklist:
                current = worklist.pop()
                if current in {header.start, exit_start} or current in region:
                    continue
                current_block = block_by_start.get(current)
                if current_block is None:
                    valid_region = False
                    break
                region.add(current)
                for successor in current_block.successors:
                    if successor in {header.start, exit_start}:
                        continue
                    if successor not in block_by_start:
                        valid_region = False
                        break
                    if successor not in region:
                        worklist.append(successor)
                if not valid_region:
                    break
            if not valid_region or not region:
                continue

            # Reject entries from outside the header. Internal cycles remain
            # explicit labels/gotos when the enclosing loop is structured; a
            # later structuring pass may recognize those cycles independently.
            if any(
                predecessor.start not in region and predecessor.start != header.start
                and any(successor in region for successor in predecessor.successors)
                for predecessor in blocks
            ):
                continue

            header_label = f"loc_{header.start:X}"
            body_label = f"loc_{loop_body_start:X}"
            exit_label = f"loc_{exit_start:X}"
            if not all(label in labels for label in (header_label, body_label, exit_label)):
                continue
            header_index = labels[header_label]
            body_index = labels[body_label]
            exit_index = labels[exit_label]
            if not (header_index < body_index < exit_index):
                continue
            region_labels = {
                f"loc_{start:X}": labels.get(f"loc_{start:X}") for start in region
            }
            if any(index is None for index in region_labels.values()):
                continue
            textual_region = sorted(
                ((labels[f"loc_{start:X}"], start) for start in region)
            )
            region_positions = [index for index, _start in textual_region]
            if not region_positions or region_positions[0] != body_index:
                continue
            if region_positions[-1] >= exit_index:
                continue
            region_end_index = exit_index
            # Every emitted label between the loop body and its textual end
            # must belong to this region.
            intervening_labels = {
                index for index in labels.values()
                if body_index <= index < region_end_index
            }
            if intervening_labels != set(region_positions):
                continue

            ordered_region = [start for _index, start in textual_region]
            # The text emitter relies on physical fallthrough between adjacent
            # blocks. Do not structure a region whose emitted order changes it.
            fallthrough_ok = True
            for position, start in enumerate(ordered_region):
                current_block = block_by_start[start]
                last = current_block.instructions[-1] if current_block.instructions else None
                if last is None or last.flow == "jump" or last.flow == "return":
                    continue
                next_text_block = ordered_region[position + 1] if position + 1 < len(ordered_region) else exit_start
                if last.flow == "conditional":
                    fallthrough = current_block.end
                    if fallthrough != next_text_block:
                        fallthrough_ok = False
                        break
                elif current_block.end != next_text_block:
                    fallthrough_ok = False
                    break
            if not fallthrough_ok:
                continue

            header_end_index = min(
                (index for index in labels.values() if index > header_index),
                default=len(body),
            )
            condition_lines = body[header_index + 1:header_end_index]
            preserve_body_label = any(
                successor == loop_body_start
                for start in region for successor in block_by_start[start].successors
            )
            body_lines = body[body_index + 1:exit_index]
            after_loop_lines = body[exit_index + 1:]
            branch_index = next(
                (
                    index for index in range(len(condition_lines) - 1, -1, -1)
                    if condition_lines[index].lstrip().startswith("if (")
                    and f"goto loc_{loop_body_start:X};" in condition_lines[index]
                ),
                None,
            )
            if branch_index is None:
                continue
            branch = condition_lines[branch_index].strip()
            match = re.fullmatch(r"if \((.+)\) goto loc_[0-9A-F]+;", branch)
            if match is None:
                continue
            loop_body_lines: list[str] = []
            for line in body_lines:
                stripped = line.strip()
                unconditional = re.fullmatch(r"goto loc_([0-9A-F]+);", stripped)
                if unconditional:
                    target = int(unconditional.group(1), 16)
                    if target == header.start:
                        loop_body_lines.append("    continue;")
                        continue
                    if target == exit_start:
                        loop_body_lines.append("    break;")
                        continue
                conditional = re.fullmatch(r"if \((.+)\) goto loc_([0-9A-F]+);", stripped)
                if conditional:
                    target = int(conditional.group(2), 16)
                    if target == header.start:
                        loop_body_lines.append(f"    if ({conditional.group(1)}) continue;")
                        continue
                    if target == exit_start:
                        loop_body_lines.append(f"    if ({conditional.group(1)}) break;")
                        continue
                loop_body_lines.append(line)
            structured = body[:header_index]
            structured.extend(condition_lines[:branch_index])
            structured.append(f"    while ({match.group(1)}) {{")
            structured.extend(f"    {line}" for line in loop_body_lines)
            structured.append("    }")
            structured.extend(after_loop_lines)
            return structured
        return body

    @staticmethod
    def _structure_simple_diamonds(body: list[str], blocks: list[BasicBlock]) -> list[str]:
        """Convert one-level, unambiguous CFG diamonds into if/else statements.

        This deliberately handles only the shape where both conditional successors
        immediately reconverge at the same block. Loops, switches, irreducible
        graphs, and nested diamonds remain in the safer label/goto representation.
        """
        block_by_start = {block.start: block for block in blocks}
        labels = {f"loc_{block.start:X}": index for index, line in enumerate(body)
                  for block in blocks if line == f"loc_{block.start:X}:"}
        for condition_block in sorted(blocks, key=lambda block: block.start):
            if not condition_block.instructions or condition_block.instructions[-1].flow != "conditional":
                continue
            successors = list(condition_block.successors)
            if len(successors) != 2:
                continue
            true_start = condition_block.instructions[-1].direct_target
            if true_start not in successors:
                continue
            false_start = next(start for start in successors if start != true_start)
            true_block = block_by_start.get(true_start)
            false_block = block_by_start.get(false_start)
            if true_block is None or false_block is None:
                continue
            true_next = true_block.successors
            false_next = false_block.successors
            if len(true_next) != 1 or len(false_next) != 1 or true_next[0] != false_next[0]:
                continue
            join_start = true_next[0]
            if join_start in {condition_block.start, true_start, false_start}:
                continue
            condition_label = f"loc_{condition_block.start:X}:"
            true_label = f"loc_{true_start:X}:"
            false_label = f"loc_{false_start:X}:"
            join_label = f"loc_{join_start:X}:"
            if not all(label in labels for label in (condition_label[:-1], true_label[:-1], false_label[:-1], join_label[:-1])):
                continue
            condition_index = labels[condition_label[:-1]]
            true_index = labels[true_label[:-1]]
            false_index = labels[false_label[:-1]]
            join_index = labels[join_label[:-1]]
            if not (condition_index < true_index < join_index and
                    condition_index < false_index < join_index):
                continue
            first_branch_index = min(true_index, false_index)
            second_branch_index = max(true_index, false_index)
            condition_lines = body[condition_index + 1:first_branch_index]
            first_branch_lines = body[first_branch_index + 1:second_branch_index]
            second_branch_lines = body[second_branch_index + 1:join_index]
            true_lines, false_lines = (
                (first_branch_lines, second_branch_lines)
                if true_index == first_branch_index
                else (second_branch_lines, first_branch_lines)
            )
            branch_line_index = next(
                (index for index in range(len(condition_lines) - 1, -1, -1)
                 if condition_lines[index].lstrip().startswith("if (") and "goto " in condition_lines[index]),
                None,
            )
            if branch_line_index is None:
                continue
            branch_line = condition_lines[branch_line_index].strip()
            match = re.fullmatch(r"if \((.+)\) goto loc_[0-9A-F]+;", branch_line)
            if match is None:
                continue
            condition = match.group(1)
            true_lines = [
                line for line in true_lines
                if not line.strip().startswith(f"goto loc_{join_start:X};")
            ]
            false_lines = [
                line for line in false_lines
                if not line.strip().startswith(f"goto loc_{join_start:X};")
            ]
            structured = body[:condition_index]
            structured.extend(condition_lines[:branch_line_index])
            structured.append(f"    if ({condition}) {{")
            structured.extend(f"    {line}" for line in true_lines)
            structured.append("    } else {")
            structured.extend(f"    {line}" for line in false_lines)
            structured.append("    }")
            structured.extend(body[join_index + 1:])
            return structured
        return body

    @staticmethod
    def _invert_condition(cond: str) -> str:
        cond = cond.strip()
        ops = [
            ("!=", "=="),
            ("==", "!="),
            ("<=", ">"),
            (">=", "<"),
            ("<", ">="),
            (">", "<="),
        ]
        for op, inv in ops:
            if f" {op} " in cond:
                return cond.replace(f" {op} ", f" {inv} ", 1)
        cond_names = {
            "condition_equal": "condition_not_equal",
            "condition_not_equal": "condition_equal",
            "condition_zero": "condition_not_zero",
            "condition_not_zero": "condition_zero",
            "condition_less": "condition_greater_or_equal",
            "condition_greater_or_equal": "condition_less",
            "condition_less_or_equal": "condition_greater",
            "condition_greater": "condition_less_or_equal",
            "condition_unsigned_below": "condition_unsigned_above_or_equal",
            "condition_unsigned_above_or_equal": "condition_unsigned_below",
            "condition_unsigned_below_or_equal": "condition_unsigned_above",
            "condition_unsigned_above": "condition_unsigned_below_or_equal",
        }
        if cond in cond_names:
            return cond_names[cond]
        if cond.startswith("!"):
            return cond[1:].strip("()")
        return f"!({cond})"

    @staticmethod
    def _structure_single_if(body: list[str], blocks: list[BasicBlock]) -> list[str]:
        block_by_start = {block.start: block for block in blocks}
        labels = {
            line.strip()[:-1]: index
            for index, line in enumerate(body)
            if line.strip().startswith("loc_") and line.strip().endswith(":")
        }

        for block in sorted(blocks, key=lambda b: b.start):
            if not block.instructions or block.instructions[-1].flow != "conditional":
                continue
            if len(block.successors) != 2:
                continue
            branch_target = block.instructions[-1].direct_target
            if branch_target not in block.successors:
                continue
            fallthrough = next(s for s in block.successors if s != branch_target)

            fall_block = block_by_start.get(fallthrough)
            if fall_block is not None and fall_block.successors == [branch_target]:
                cond_lbl = f"loc_{block.start:X}"
                then_lbl = f"loc_{fallthrough:X}"
                join_lbl = f"loc_{branch_target:X}"

                if cond_lbl in labels and then_lbl in labels and join_lbl in labels:
                    c_idx, t_idx, j_idx = labels[cond_lbl], labels[then_lbl], labels[join_lbl]
                    if c_idx < t_idx < j_idx:
                        cond_lines = body[c_idx + 1:t_idx]
                        then_lines = body[t_idx + 1:j_idx]

                        br_idx = next(
                            (idx for idx in range(len(cond_lines) - 1, -1, -1)
                             if "if (" in cond_lines[idx] and f"goto {join_lbl};" in cond_lines[idx]),
                            None,
                        )
                        if br_idx is not None:
                            br_line = cond_lines[br_idx].strip()
                            m = re.fullmatch(rf"if \((.+)\) goto {join_lbl};", br_line)
                            if m:
                                cond_expr = m.group(1)
                                inv_cond = PseudocodeLifter._invert_condition(cond_expr)
                                then_clean = [
                                    l for l in then_lines
                                    if not l.strip().startswith(f"goto {join_lbl};")
                                ]

                                res = body[:c_idx]
                                res.extend(cond_lines[:br_idx])
                                res.append(f"    if ({inv_cond}) {{")
                                for l in then_clean:
                                    stripped = l.strip()
                                    if stripped:
                                        res.append(f"        {stripped}")
                                res.append("    }")
                                res.extend(body[j_idx:])
                                return res

            target_block = block_by_start.get(branch_target)
            if target_block is not None and target_block.successors == [fallthrough]:
                cond_lbl = f"loc_{block.start:X}"
                then_lbl = f"loc_{branch_target:X}"
                join_lbl = f"loc_{fallthrough:X}"

                if cond_lbl in labels and then_lbl in labels and join_lbl in labels:
                    c_idx, t_idx, j_idx = labels[cond_lbl], labels[then_lbl], labels[join_lbl]
                    if c_idx < t_idx < j_idx:
                        cond_lines = body[c_idx + 1:t_idx]
                        then_lines = body[t_idx + 1:j_idx]

                        br_idx = next(
                            (idx for idx in range(len(cond_lines) - 1, -1, -1)
                             if "if (" in cond_lines[idx] and f"goto {then_lbl};" in cond_lines[idx]),
                            None,
                        )
                        if br_idx is not None:
                            br_line = cond_lines[br_idx].strip()
                            m = re.fullmatch(rf"if \((.+)\) goto {then_lbl};", br_line)
                            if m:
                                cond_expr = m.group(1)
                                then_clean = [
                                    l for l in then_lines
                                    if not l.strip().startswith(f"goto {join_lbl};")
                                ]

                                res = body[:c_idx]
                                res.extend(cond_lines[:br_idx])
                                res.append(f"    if ({cond_expr}) {{")
                                for l in then_clean:
                                    stripped = l.strip()
                                    if stripped:
                                        res.append(f"        {stripped}")
                                res.append("    }")
                                res.extend(body[j_idx:])
                                return res
        return body

    @staticmethod
    def _clean_compiler_comments(body: list[str]) -> list[str]:
        cleaned = []
        for line in body:
            s = line.strip()
            if s.startswith("/* condition flags set by"):
                continue
            if s.startswith("/* machine-level stack operation:"):
                continue
            if s.startswith("/* stack pointer adjustment:"):
                continue
            cleaned.append(line)
        return cleaned

    @staticmethod
    def _remove_unreferenced_labels(body: list[str]) -> list[str]:
        text = "\n".join(body)
        referenced = set(re.findall(r"\bgoto\s+(loc_[0-9A-Fa-f]+);", text))
        referenced.update(re.findall(r"\bgoto\s+([A-Za-z0-9_]+);", text))
        cleaned = []
        for line in body:
            s = line.strip()
            if s.startswith("loc_") and s.endswith(":"):
                lbl = s[:-1]
                if lbl not in referenced:
                    continue
            cleaned.append(line)
        return cleaned

    @staticmethod
    def _prettify_c_syntax(code: str) -> str:
        code = re.sub(r"\*\(\(\(([a-zA-Z0-9_]+)\s*\*\)\(([a-zA-Z0-9_]+)\)\)\)", r"*(\1 *)\2", code)
        code = re.sub(r"\*\(\(\(([a-zA-Z0-9_]+)\s*\*\)\((.+?)\)\)\)", r"*(\1 *)(\2)", code)
        return code
