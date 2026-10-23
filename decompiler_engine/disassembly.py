from __future__ import annotations
import re
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64, CS_OP_IMM, CS_OP_MEM
from capstone.x86_const import X86_REG_RIP
from .models import Instruction
from .pe_loader import PEImage

CONDITIONAL_PREFIXES = ("ja", "jb", "jc", "je", "jg", "jl", "jna", "jnb", "jnc", "jne", "jng", "jnl", "jno", "jnp", "jns", "jnz", "jo", "jp", "jpe", "jpo", "js", "jz", "loop", "jecxz", "jrcxz", "jcxz")

def classify_flow(mnemonic: str) -> str:
    m = mnemonic.lower()
    if m in {"call", "lcall", "callq"}: return "call"
    if m.startswith("ret") or m in {"iret", "iretd", "iretq"}: return "return"
    if m in {"jmp", "ljmp", "jmpq"}: return "jump"
    if m.startswith(CONDITIONAL_PREFIXES) or (m.startswith("j") and m != "jmp"):
        return "conditional"
    return "normal"

class Disassembler:
    """Single-pass linear-sweep decoder. Fast broad scan; embedded data can decode as instructions."""
    def __init__(self, image: PEImage, max_instructions: int = 2_000_000):
        self.image = image
        mode = CS_MODE_64 if image.bits == 64 else CS_MODE_32
        self.md = Cs(CS_ARCH_X86, mode)
        self.md.detail = True
        # Continue scanning after invalid bytes/alignment data rather than silently
        # abandoning the rest of an executable section. Such bytes are reported as .byte.
        self.md.skipdata = True
        self.max_instructions = max_instructions
        self.warnings: list[str] = ["Decoder skip-data mode is enabled; .byte records are placeholders, not confirmed instructions."]

    def run(self) -> list[Instruction]:
        output: list[Instruction] = []
        for sec in self.image.sections:
            if not sec.executable:
                continue
            blob = self.image.section_bytes(sec)
            if not blob:
                continue
            for ins in self.md.disasm(blob, sec.start):
                if len(output) >= self.max_instructions:
                    self.warnings.append(f"Instruction cap reached ({self.max_instructions:,}); analysis is partial.")
                    return output
                m = ins.mnemonic.lower()
                direct_target = None
                reference_target = None
                indirect_targets: list[int] | None = None
                jump_table_entry_size = None
                jump_table_base = None
                jump_table_entries_observed = None
                jump_table_slots = None
                indirect_call_targets: list[int] | None = None
                try:
                    if ins.operands:
                        op0 = ins.operands[0]
                        if op0.type == CS_OP_IMM and classify_flow(m) in ("call", "jump", "conditional"):
                            direct_target = int(op0.imm)
                        # Generic RIP-relative data/code reference, including import IAT call stubs.
                        for op in ins.operands:
                            if op.type == CS_OP_MEM:
                                if op.mem.base == X86_REG_RIP:
                                    reference_target = int(ins.address + ins.size + op.mem.disp)
                                    break
                                # x86 absolute memory operands often use no base/index.
                                if not op.mem.base and not op.mem.index and op.mem.disp:
                                    reference_target = int(op.mem.disp)
                                    break
                except Exception:
                    pass
                if classify_flow(m) == "jump" and direct_target is None and reference_target is not None:
                    (indirect_targets, jump_table_entry_size,
                     jump_table_entries_observed, jump_table_slots) = self._read_jump_table_slots(
                        reference_target
                    )
                    jump_table_slots = jump_table_slots or None
                    jump_table_base = reference_target if indirect_targets else None
                if classify_flow(m) == "call" and direct_target is None and reference_target in self.image.exports:
                    indirect_call_targets = [reference_target]
                resolved_call_provenance = (
                    "export_address" if indirect_call_targets else None
                )
                import_name = self.image.imports.get(reference_target) if reference_target is not None else None
                output.append(Instruction(
                    address=int(ins.address), size=int(ins.size), bytes_hex=ins.bytes.hex(" ").upper(),
                    mnemonic=ins.mnemonic, operands=ins.op_str, section=sec.name,
                    direct_target=direct_target, reference_target=reference_target,
                    flow=classify_flow(m), import_name=import_name, indirect_targets=indirect_targets,
                    jump_table_entry_size=jump_table_entry_size, jump_table_base=jump_table_base,
                    indirect_call_targets=indirect_call_targets,
                    resolved_call_provenance=resolved_call_provenance,
                    jump_table_entries_observed=jump_table_entries_observed,
                    jump_table_slots=jump_table_slots))
        output.sort(key=lambda x: x.address)
        for index, ins in enumerate(output):
            if ins.flow not in {"jump", "call"} or ins.direct_target is not None or ins.reference_target is not None:
                continue
            table_base = self._resolve_jump_table_base(output, index)
            if table_base is not None:
                ins.reference_target = table_base
            else:
                relative = self._resolve_relative_register_table(output, index)
                if relative is not None:
                    ins.reference_target, ins.jump_table_index_register = relative
        for index, ins in enumerate(output):
            if ins.flow not in {"jump", "call"} or ins.direct_target is not None:
                continue
            bound = self._infer_jump_table_bound(
                output, index, ins.jump_table_index_register
            )
            if bound is None:
                continue
            ins.jump_table_index_register = bound["index_register"]
            ins.jump_table_case_count = bound["case_count"]
            ins.jump_table_bound_branch = bound["branch"]
            ins.jump_table_bound_instruction = bound["compare_address"]
            ins.jump_table_index_bias = bound["index_bias"]
            ins.jump_table_default_target = bound["default_target"]
            if ins.reference_target is not None and bound["case_count"] > 0:
                targets, entry_size, observed, slots = self._read_jump_table_slots(
                    ins.reference_target, min(bound["case_count"], 256), allow_sparse=True
                )
                if ins.flow == "call":
                    instruction_starts = {item.address for item in output}
                    ins.indirect_call_targets = sorted({
                        target for target in targets if target in instruction_starts
                    }) or None
                    if ins.indirect_call_targets:
                        ins.resolved_call_provenance = "bounded_call_table"
                    slots = [target if target in instruction_starts else None for target in slots]
                else:
                    ins.indirect_targets = targets
                ins.jump_table_entry_size = entry_size
                ins.jump_table_entries_observed = observed
                ins.jump_table_base = ins.reference_target if targets else None
                ins.jump_table_slots = slots
                ins.jump_table_default_entry_indices = self._jump_table_default_entry_indices(
                    slots, ins.jump_table_default_target
                )
        if not output:
            self.warnings.append("No instructions decoded from executable sections.")
        return output

    def run_recursive(
        self, seeds: set[int], candidates: list[Instruction] | None = None,
        existing_instructions: list[Instruction] | None = None,
        priority_seeds: set[int] | None = None,
    ) -> list[Instruction]:
        """Decode instructions reached from trusted entries and control flow.

        This pass deliberately uses skipdata=False. The linear sweep remains useful
        for finding possible code and jump-table data, but its boundaries are not
        admitted to the CFG unless reachable from a seed through decoded flow.
        """
        decoder = Cs(CS_ARCH_X86, CS_MODE_64 if self.image.bits == 64 else CS_MODE_32)
        decoder.detail = True
        decoder.skipdata = False
        sections = [sec for sec in self.image.sections if sec.executable]
        blobs = {sec.start: self.image.section_bytes(sec) for sec in sections}
        candidate_by_addr = {ins.address: ins for ins in (candidates or [])}
        priority = set(priority_seeds or ()) & set(seeds)
        # The worklist is popped from the end. Put application roots last so they
        # are decoded before metadata-only seeds when the instruction cap is tight.
        pending = sorted(set(seeds) - priority, reverse=True)
        pending.extend(sorted(priority, reverse=True))
        decoded: dict[int, Instruction] = {
            ins.address: ins for ins in (existing_instructions or [])
        }
        occupied: dict[int, int] = {
            pos: ins.address
            for ins in decoded.values()
            for pos in range(ins.address + 1, ins.end)
        }

        def section_at(address: int):
            return next((sec for sec in sections if sec.start <= address < sec.end), None)

        while pending and len(decoded) < self.max_instructions:
            address = pending.pop()
            while len(decoded) < self.max_instructions:
                sec = section_at(address)
                if sec is None or address in decoded or address in occupied:
                    break
                blob = blobs.get(sec.start, b"")
                offset = address - sec.start
                if offset < 0 or offset >= len(blob):
                    break
                ins_iter = decoder.disasm(blob[offset:], address, count=1)
                ins = next(ins_iter, None)
                if ins is None or ins.size <= 0 or ins.address != address:
                    break
                end = int(ins.address + ins.size)
                if any(pos in occupied for pos in range(address + 1, end)):
                    break
                m = ins.mnemonic.lower()
                flow = classify_flow(m)
                direct_target = None
                reference_target = None
                try:
                    if ins.operands:
                        op0 = ins.operands[0]
                        if op0.type == CS_OP_IMM and flow in {"call", "jump", "conditional"}:
                            direct_target = int(op0.imm)
                        for op in ins.operands:
                            if op.type == CS_OP_MEM and op.mem.base == X86_REG_RIP:
                                reference_target = int(ins.address + ins.size + op.mem.disp)
                                break
                            if (op.type == CS_OP_MEM and not op.mem.base and
                                    not op.mem.index and op.mem.disp):
                                reference_target = int(op.mem.disp)
                                break
                except Exception:
                    pass
                item = Instruction(
                    address=int(ins.address), size=int(ins.size),
                    bytes_hex=ins.bytes.hex(" ").upper(), mnemonic=ins.mnemonic,
                    operands=ins.op_str, section=sec.name, direct_target=direct_target,
                    reference_target=reference_target, flow=flow,
                    import_name=(self.image.imports.get(reference_target)
                                 if reference_target is not None else None),
                    discovery_source="recursive_control_flow",
                    boundary_confidence="confirmed_reachable",
                )
                # Retain richer jump-table/call evidence from the broad candidate pass
                # only when it describes this exact, independently decoded boundary.
                candidate = candidate_by_addr.get(item.address)
                if candidate is not None and candidate.size == item.size:
                    for name in (
                        "indirect_targets", "jump_table_entry_size", "jump_table_base",
                        "indirect_call_targets", "resolved_call_provenance",
                        "jump_table_entries_observed", "jump_table_index_register",
                        "jump_table_case_count", "jump_table_bound_branch",
                        "jump_table_bound_instruction", "jump_table_index_bias",
                        "jump_table_default_target", "jump_table_default_entry_indices",
                        "jump_table_slots", "resolved_indirect_jump_target",
                        "unresolved_call_reason",
                    ):
                        setattr(item, name, getattr(candidate, name))
                decoded[item.address] = item
                for pos in range(item.address + 1, end):
                    occupied[pos] = item.address
                if direct_target is not None and flow in {"call", "jump", "conditional"}:
                    if section_at(direct_target) is not None:
                        pending.append(direct_target)
                for target in item.indirect_targets or ():
                    if section_at(target) is not None:
                        pending.append(target)
                for target in item.indirect_call_targets or ():
                    if section_at(target) is not None:
                        pending.append(target)
                if flow in {"return", "jump"}:
                    break
                address = end
        if len(decoded) >= self.max_instructions and pending:
            self.warnings.append(
                f"Recursive decode instruction cap reached ({self.max_instructions:,}); analysis is partial."
            )
        return sorted(decoded.values(), key=lambda item: item.address)

    @staticmethod
    def _register_family(register: str) -> str:
        register = register.lower()
        for aliases in (
            ("rax", "eax", "ax", "al", "ah"),
            ("rbx", "ebx", "bx", "bl", "bh"),
            ("rcx", "ecx", "cx", "cl", "ch"),
            ("rdx", "edx", "dx", "dl", "dh"),
            ("rsi", "esi", "si", "sil"),
            ("rdi", "edi", "di", "dil"),
            ("rbp", "ebp", "bp", "bpl"),
            ("rsp", "esp", "sp", "spl"),
        ):
            if register in aliases:
                return aliases[0]
        match = re.fullmatch(r"r(8|9|1[0-5])(?:d|w|b)?", register)
        return f"r{match.group(1)}" if match else register

    @staticmethod
    def _jump_table_index_register(operands: str) -> str | None:
        memory = re.search(r"\[([^\]]+)\]", operands)
        if not memory:
            return None
        scaled = re.search(r"\b([a-z][a-z0-9]*)\s*\*\s*(?:2|4|8)\b", memory.group(1), re.I)
        if not scaled:
            return None
        return scaled.group(1).lower()

    @classmethod
    def _jump_table_base_register(cls, operands: str) -> str | None:
        memory = re.search(r"\[([^\]]+)\]", operands)
        if not memory:
            return None
        expression = memory.group(1)
        scaled = re.search(r"\b([a-z][a-z0-9]*)\s*\*\s*(?:2|4|8)\b", expression, re.I)
        if not scaled:
            return None
        expression = expression[:scaled.start()] + " " + expression[scaled.end():]
        expression = re.sub(r"(?:0x[0-9a-f]+|\d+)", " ", expression, flags=re.I)
        registers = [
            token.lower() for token in re.findall(r"\b[a-z][a-z0-9]*\b", expression, re.I)
            if token.lower() not in {"byte", "word", "dword", "qword", "ptr"}
        ]
        if len(registers) != 1:
            return None
        return registers[0]

    @classmethod
    def _resolve_jump_table_base(cls, instructions: list[Instruction], jump_index: int) -> int | None:
        """Resolve a bounded indexed memory-jump base from its nearest definition."""
        if not 0 <= jump_index < len(instructions):
            return None
        jump = instructions[jump_index]
        if cls._jump_table_index_register(jump.operands) is None:
            return None
        base_register = cls._jump_table_base_register(jump.operands)
        if base_register is None:
            return None
        base_family = cls._register_family(base_register)
        for prior in reversed(instructions[max(0, jump_index - 16):jump_index]):
            if prior.section != jump.section:
                return None
            if prior.flow in {"call", "jump", "return"}:
                return None
            operands = [part.strip() for part in prior.operands.split(",")]
            if not operands:
                continue
            destination = re.fullmatch(r"([a-z][a-z0-9]*)", operands[0], re.I)
            if not destination or cls._register_family(destination.group(1)) != base_family:
                continue
            if destination.group(1).lower() != base_register.lower():
                # Avoid treating zero-extending/width-changing aliases as the
                # same pointer value (for example, EAX versus RAX).
                return None
            mnemonic = prior.mnemonic.lower()
            if mnemonic == "lea" and prior.reference_target is not None:
                table_expression = re.search(r"\[([^\]]+)\]", jump.operands)
                if not table_expression:
                    return None
                displacement_expression = table_expression.group(1)
                scaled = re.search(
                    r"\b([a-z][a-z0-9]*)\s*\*\s*(?:2|4|8)\b",
                    displacement_expression, re.I,
                )
                if scaled:
                    displacement_expression = (
                        displacement_expression[:scaled.start()] + " "
                        + displacement_expression[scaled.end():]
                    )
                displacement_expression = re.sub(
                    rf"\b{re.escape(base_register)}\b", " ", displacement_expression,
                    count=1, flags=re.I,
                )
                displacement = 0
                for term in re.finditer(r"([+-]?)\s*(0x[0-9a-f]+|\d+)", displacement_expression, re.I):
                    text = term.group(2)
                    amount = int(text, 16 if text.lower().startswith("0x") else 10)
                    displacement += -amount if term.group(1) == "-" else amount
                return prior.reference_target + displacement
            if mnemonic in {"mov", "movabs"} and len(operands) == 2:
                immediate = re.fullmatch(r"(?:0x[0-9a-f]+|\d+)", operands[1], re.I)
                if immediate:
                    return int(immediate.group(0), 16 if immediate.group(0).lower().startswith("0x") else 10)
            # The first write to the base register is unsupported, so an older
            # definition cannot be trusted to reach this indirect jump.
            return None
        return None

    @classmethod
    def _resolve_relative_register_table(
        cls, instructions: list[Instruction], jump_index: int
    ) -> tuple[int, str] | None:
        """Recognize `movsxd tmp,[table+index*4]; add tmp,table; jmp tmp`."""
        jump = instructions[jump_index]
        target = re.fullmatch(r"([a-z][a-z0-9]*)", jump.operands.strip(), re.I)
        if not target:
            return None
        target_family = cls._register_family(target.group(1))
        start = max(0, jump_index - 8)
        window = instructions[start:jump_index]
        for add_pos in range(len(window) - 1, -1, -1):
            add = window[add_pos]
            parts = [part.strip() for part in add.operands.split(",")]
            if (add.mnemonic.lower() not in {"add", "lea"} or len(parts) != 2
                    or cls._register_family(parts[0]) != target_family):
                continue
            if add.mnemonic.lower() == "add":
                base_match = re.fullmatch(r"([a-z][a-z0-9]*)", parts[1], re.I)
            else:
                base_match = re.search(r"\+\s*([a-z][a-z0-9]*)\s*\]", parts[1], re.I)
            if not base_match:
                continue
            base_register = base_match.group(1).lower()
            base_family = cls._register_family(base_register)
            for load in reversed(window[:add_pos]):
                if load.flow != "normal" or load.section != jump.section:
                    break
                load_parts = [part.strip() for part in load.operands.split(",", 1)]
                if len(load_parts) != 2 or cls._register_family(load_parts[0]) != target_family:
                    continue
                if load.mnemonic.lower() not in {"movsxd", "movsx", "mov"}:
                    return None
                memory = re.search(r"\[([^\]]+)\]", load_parts[1])
                if not memory:
                    return None
                scaled = re.search(r"\b([a-z][a-z0-9]*)\s*\*\s*4\b", memory.group(1), re.I)
                if not scaled or not re.search(rf"\b{re.escape(base_register)}\b", memory.group(1), re.I):
                    return None
                index_register = scaled.group(1).lower()
                for prior in reversed(window[:window.index(load)]):
                    if prior.flow in {"call", "jump", "return"}:
                        break
                    prior_parts = [part.strip() for part in prior.operands.split(",")]
                    if not prior_parts or not re.fullmatch(r"[a-z][a-z0-9]*", prior_parts[0], re.I):
                        continue
                    if cls._register_family(prior_parts[0]) != base_family:
                        continue
                    if prior.mnemonic.lower() == "lea" and prior.reference_target is not None:
                        if prior_parts[0].lower() != base_register:
                            return None
                        return prior.reference_target, index_register
                    return None
        return None

    @staticmethod
    def _jump_table_default_entry_indices(
        targets: list[int | None], default_target: int | None
    ) -> list[int]:
        if default_target is None:
            return []
        return [index for index, target in enumerate(targets) if target == default_target]

    @classmethod
    def _has_nonnegative_index_guard(
        cls, instructions: list[Instruction], index_family: str, default_target: int
    ) -> bool:
        """Require a matching signed lower-bound guard before signed upper tests."""
        for branch_index in range(len(instructions) - 1, -1, -1):
            branch = instructions[branch_index]
            if (branch.flow != "conditional" or branch.direct_target != default_target
                    or branch.mnemonic.lower() not in {"js", "jl", "jnge"}):
                continue
            for check_index in range(branch_index - 1, max(-1, branch_index - 5), -1):
                check = instructions[check_index]
                if check.flow in {"call", "jump", "return"} or check.flow == "conditional":
                    break
                operands = [part.strip() for part in check.operands.split(",")]
                if len(operands) != 2:
                    continue
                left = re.fullmatch(r"(?:[a-z]+\s+ptr\s+)?([a-z][a-z0-9]*)", operands[0], re.I)
                right_reg = re.fullmatch(r"(?:[a-z]+\s+ptr\s+)?([a-z][a-z0-9]*)", operands[1], re.I)
                immediate = re.fullmatch(r"(?:0x[0-9a-f]+|\d+)", operands[1], re.I)
                if not left or cls._register_family(left.group(1)) != index_family:
                    continue
                mnemonic = check.mnemonic.lower()
                if mnemonic == "test" and right_reg and cls._register_family(right_reg.group(1)) == index_family:
                    return True
                if mnemonic == "cmp" and immediate:
                    value = int(immediate.group(0), 16 if immediate.group(0).lower().startswith("0x") else 10)
                    if value == 0:
                        return True
        return False

    @classmethod
    def _infer_jump_table_bound(cls, instructions: list[Instruction], jump_index: int,
                                index_override: str | None = None) -> dict[str, int | str] | None:
        """Match an indexed indirect jump to a nearby unsigned upper-bound guard."""
        if not 0 <= jump_index < len(instructions):
            return None
        jump = instructions[jump_index]
        index_register = index_override or cls._jump_table_index_register(jump.operands)
        if index_register is None:
            return None
        index_family = cls._register_family(index_register)
        window_start = max(0, jump_index - 8)
        window = instructions[window_start:jump_index]
        branch_cases = {
            "ja": ("inclusive", "ja"), "jnbe": ("inclusive", "jnbe"),
            "jae": ("exclusive", "jae"), "jnb": ("exclusive", "jnb"), "jnc": ("exclusive", "jnc"),
            "jg": ("signed_inclusive", "jg"), "jnle": ("signed_inclusive", "jnle"),
            "jge": ("signed_exclusive", "jge"), "jnl": ("signed_exclusive", "jnl"),
        }
        for branch_local_index in range(len(window) - 1, -1, -1):
            branch = window[branch_local_index]
            branch_mnemonic = branch.mnemonic.lower()
            if branch.flow != "conditional" or branch_mnemonic not in branch_cases or branch.direct_target is None:
                continue
            between = window[branch_local_index + 1:]
            if any(item.flow in {"call", "jump", "conditional", "return"} for item in between):
                continue
            if branch.direct_target == jump.address or any(item.address == branch.direct_target for item in between):
                continue
            for compare_local_index in range(branch_local_index - 1, -1, -1):
                compare = window[compare_local_index]
                if compare.mnemonic.lower() != "cmp":
                    continue
                compare_to_jump = window[compare_local_index:]
                if any(left.end != right.address or left.section != right.section
                       for left, right in zip(compare_to_jump, compare_to_jump[1:])):
                    continue
                if any(item.flow in {"jump", "return"}
                       or (item.flow == "call" and item is not jump)
                       or (item.flow == "conditional" and item is not branch)
                       for item in compare_to_jump):
                    continue
                operands = [part.strip() for part in compare.operands.split(",", 1)]
                if len(operands) != 2:
                    continue
                register_match = re.fullmatch(r"(?:[a-z]+\s+ptr\s+)?([a-z][a-z0-9]*)", operands[0], re.I)
                immediate_match = re.fullmatch(r"(?:0x[0-9a-f]+|\d+)", operands[1], re.I)
                if not register_match or not immediate_match:
                    continue
                if cls._register_family(register_match.group(1)) != index_family:
                    continue
                immediate_text = immediate_match.group(0)
                value = int(immediate_text, 16) if immediate_text.lower().startswith("0x") else int(immediate_text, 10)
                if value < 0 or value > 0x100000:
                    continue
                kind, branch_name = branch_cases[branch_mnemonic]
                if kind.startswith("signed_") and not cls._has_nonnegative_index_guard(
                    window[:compare_local_index], index_family, branch.direct_target
                ):
                    continue
                case_count = value + 1 if kind in {"inclusive", "signed_inclusive"} else value
                index_bias = 0
                if compare_local_index > 0:
                    adjust = window[compare_local_index - 1]
                    if adjust.end == compare.address and adjust.section == compare.section:
                        adjust_operands = [part.strip() for part in adjust.operands.split(",", 1)]
                        adjust_mnemonic = adjust.mnemonic.lower()
                        if len(adjust_operands) == 2 and cls._register_family(adjust_operands[0]) == index_family:
                            immediate = re.fullmatch(r"[-+]?(?:0x[0-9a-f]+|\d+)", adjust_operands[1], re.I)
                            if immediate and adjust_mnemonic in {"sub", "add"}:
                                immediate_text = immediate.group(0)
                                unsigned_text = immediate_text.lstrip("+-")
                                amount = int(immediate_text, 16) if unsigned_text.lower().startswith("0x") else int(immediate_text, 10)
                                index_bias = amount if adjust_mnemonic == "sub" else -amount
                            elif adjust_mnemonic == "lea":
                                address = re.fullmatch(
                                    r"\[\s*([a-z][a-z0-9]*)\s*([+-])\s*(0x[0-9a-f]+|\d+)\s*\]",
                                    adjust_operands[1], re.I,
                                )
                                if address and cls._register_family(address.group(1)) == index_family:
                                    amount_text = address.group(3)
                                    amount = int(amount_text, 16) if amount_text.lower().startswith("0x") else int(amount_text, 10)
                                    index_bias = amount if address.group(2) == "-" else -amount
                        if abs(index_bias) > 0x100000:
                            index_bias = 0
                return {
                    "index_register": index_register,
                    "case_count": case_count,
                    "index_bias": index_bias,
                    "branch": branch_name,
                    "default_target": branch.direct_target,
                    "compare_address": compare.address,
                }
        return None

    def _read_jump_table(self, table_address: int, max_entries: int = 256) -> tuple[list[int], int | None]:
        """Backward-compatible target/width view of the bounded table scan."""
        targets, entry_size, _observed = self._read_jump_table_details(table_address, max_entries)
        return targets, entry_size

    def _read_jump_table_details(
        self, table_address: int, max_entries: int = 256
    ) -> tuple[list[int], int | None, int]:
        """Return distinct targets, entry width and contiguous valid entries observed.

        The observed count is evidence, not a proven table length: exact bounds
        require switch-index and range-check recovery that this decoder does not
        yet perform. Stop at the first invalid slot so adjacent data cannot be
        searched for a later sequence of executable-looking values.
        """
        targets, entry_size, observed, _slots = self._read_jump_table_slots(
            table_address, max_entries, allow_sparse=False
        )
        return targets, entry_size, observed

    def _read_jump_table_slots(
        self, table_address: int, max_entries: int = 256, *, allow_sparse: bool = False
    ) -> tuple[list[int], int | None, int, list[int | None]]:
        """Read bounded table slots, optionally retaining invalid holes as None.

        Sparse scanning is only used after a nearby unsigned guard supplies a
        case-count bound, so scanning across a hole cannot run into unbounded
        adjacent data. CFG edges use only executable targets; slots retain case
        positions for reporting.
        """
        entry_sizes = [8, 4] if self.image.bits == 64 else [4]
        best_targets: list[int] = []
        best_entry_size: int | None = None
        best_observed = 0
        best_slots: list[int | None] = []
        for entry_size in entry_sizes:
            raw = self.image.read_va(table_address, max_entries * entry_size)
            if len(raw) < entry_size:
                continue
            targets: list[int] = []
            slots: list[int | None] = []
            observed = 0
            for offset in range(0, len(raw) - entry_size + 1, entry_size):
                chunk = raw[offset:offset + entry_size]
                unsigned_value = int.from_bytes(chunk, "little", signed=False)
                signed_value = int.from_bytes(chunk, "little", signed=True)
                candidates = [unsigned_value]
                if entry_size == 4:
                    # MSVC/clang x64 switch tables commonly store signed offsets
                    # relative to the table or image rather than full pointers.
                    candidates.extend([
                        table_address + signed_value,
                        self.image.image_base + signed_value,
                        self.image.image_base + unsigned_value,
                    ])
                else:
                    candidates.append(table_address + signed_value)
                target = next(
                    (
                        candidate for candidate in candidates
                        if self.image.section_for(candidate) is not None
                        and self.image.section_for(candidate).executable
                    ),
                    None,
                )
                if target is None:
                    if not allow_sparse:
                        break
                    slots.append(None)
                else:
                    observed += 1
                    slots.append(target)
                    if target not in targets:
                        targets.append(target)
            if (observed, len(targets)) > (best_observed, len(best_targets)):
                best_targets = targets
                best_entry_size = entry_size
                best_observed = observed
                best_slots = slots
        return best_targets, best_entry_size, best_observed, best_slots
