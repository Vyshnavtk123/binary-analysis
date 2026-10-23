from __future__ import annotations
import csv, json, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .pe_loader import PEImage, BinaryString
from .disassembly import Disassembler
from .cfg import ControlFlowGraph
from .models import Instruction, FunctionInfo
from .pseudocode import PseudocodeLifter
from .dataflow import FunctionDataFlow, split_operands
from .ir import lift_blocks
from .ssa import construct_ssa
from .crypto_analysis import analyze_crypto, build_crypto_cbom

class DecompilerPipeline:
    def __init__(self, input_path: str, output_dir: str, max_instructions: int = 2_000_000,
                 max_functions: int = 2_000, max_pseudocode_functions: int = 500,
                 max_blocks_per_function: int = 1_000, max_pseudocode_instructions: int = 200_000,
                 min_string_length: int = 4,
                 write_assembly: bool = True, write_ir: bool = True,
                 write_dataflow: bool = True, max_dataflow_instructions: int = 200_000,
                 max_dataflow_functions: int = 500,
                 progress_callback: Callable[[dict[str, Any]], None] | None = None,
                 cancel_check: Callable[[], bool] | None = None,
                 write_ssa: bool = True):
        self.input_path = str(Path(input_path).resolve())
        self.output_dir = Path(output_dir).resolve()
        self.max_instructions = max_instructions
        self.max_functions = max_functions
        self.max_pseudocode_functions = max_pseudocode_functions
        self.max_blocks_per_function = max_blocks_per_function
        self.max_pseudocode_instructions = max_pseudocode_instructions
        self.min_string_length = min_string_length
        self.write_assembly = write_assembly
        self.write_ir = write_ir
        self.write_ssa = write_ssa
        self.write_dataflow = write_dataflow
        self.max_dataflow_instructions = max_dataflow_instructions
        self.max_dataflow_functions = max_dataflow_functions
        self.progress_callback = progress_callback
        self.cancel_check = cancel_check
        self.cancelled = False
        self.cancelled_stage: str | None = None
        self.metrics: dict[str, Any] = {}

    def _progress(self, stage: str, completed: int, total: int) -> bool:
        if self.progress_callback is not None:
            self.progress_callback({
                "stage": stage, "completed": completed, "total": total,
                "cancelled": self.cancelled,
            })
        if not self.cancelled and self.cancel_check is not None and self.cancel_check():
            self.cancelled = True
            self.cancelled_stage = stage
            if self.progress_callback is not None:
                self.progress_callback({
                    "stage": stage, "completed": completed, "total": total,
                    "cancelled": True,
                })
        return not self.cancelled

    @staticmethod
    def _json_write(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _infer_function_arity(blocks: list[Any], bits: int) -> int | None:
        """Heuristically infer Windows x64 integer parameter count from read-before-write registers."""
        if bits != 64:
            return None
        aliases = {
            "rcx": ("rcx", "ecx", "cx", "cl", "ch"),
            "rdx": ("rdx", "edx", "dx", "dl", "dh"),
            "r8": ("r8", "r8d", "r8w", "r8b"),
            "r9": ("r9", "r9d", "r9w", "r9b"),
        }
        used: set[str] = set()
        written: set[str] = set()
        register_names = {alias: family for family, values in aliases.items() for alias in values}
        import re
        token_pattern = re.compile(r"(?<![a-z0-9_])(" + "|".join(sorted(register_names, key=len, reverse=True)) + r")(?![a-z0-9_])", re.I)
        for block in sorted(blocks, key=lambda b: b.start):
            for ins in block.instructions:
                m = ins.mnemonic.lower()
                operands = split_operands(ins.operands)
                reads_text = ins.operands
                # MOV/LEA to a plain register writes its first operand; that operand is not a read.
                if m in {"mov", "lea", "movzx", "movsx", "movsxd", "pop"} and operands:
                    first = operands[0].strip().lower()
                    if first in register_names:
                        reads_text = ",".join(operands[1:])
                for alias in token_pattern.findall(reads_text):
                    fam = register_names.get(alias.lower())
                    if fam and fam not in written:
                        used.add(fam)
                # Any plain-register destination is considered written after processing reads.
                if operands:
                    first = operands[0].strip().lower()
                    fam = register_names.get(first)
                    if fam and m not in {"cmp", "test", "push", "call", "jmp"} and not m.startswith("j"):
                        written.add(fam)
        ordering = ("rcx", "rdx", "r8", "r9")
        highest = max((ordering.index(x) for x in used), default=-1)
        return highest + 1 if highest >= 0 else 0

    @staticmethod
    def _has_function_prologue(address: int, instructions: dict[int, Instruction], bits: int) -> bool:
        """Recognize a narrow x86/x64 entry-prologue shape for table dispatches."""
        first = instructions.get(address)
        if first is None:
            return False
        sequence = [first]
        cursor = first.end
        for _ in range(3):
            next_ins = instructions.get(cursor)
            if next_ins is None or next_ins.section != first.section:
                break
            sequence.append(next_ins)
            cursor = next_ins.end
        index = 0
        while index < len(sequence) and sequence[index].mnemonic.lower() in {"endbr32", "endbr64"}:
            index += 1
        body = sequence[index:]
        if not body:
            return False
        mnemonic = body[0].mnemonic.lower()
        operands = body[0].operands.replace(" ", "").lower()
        if bits == 64:
            if mnemonic == "push" and operands in {"rbp", "rdi", "rsi", "rbx"}:
                return len(body) > 1 and (
                    (body[1].mnemonic.lower() == "mov" and body[1].operands.replace(" ", "").lower() == "rbp,rsp")
                    or (body[1].mnemonic.lower() == "sub" and body[1].operands.replace(" ", "").lower().startswith("rsp,"))
                )
            return mnemonic == "sub" and operands.startswith("rsp,")
        if mnemonic == "push" and operands == "ebp":
            return len(body) > 1 and (
                (body[1].mnemonic.lower() == "mov" and body[1].operands.replace(" ", "").lower() == "ebp,esp")
                or (body[1].mnemonic.lower() == "sub" and body[1].operands.replace(" ", "").lower().startswith("esp,"))
            )
        return mnemonic == "sub" and operands.startswith("esp,")

    def run(self) -> dict[str, Any]:
        total_start = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stage = time.perf_counter()
        image = PEImage(self.input_path)
        load_seconds = time.perf_counter() - stage
        try:
            stage = time.perf_counter()
            strings = image.scan_strings(self.min_string_length, include_executable=False)
            string_scan_seconds = time.perf_counter() - stage
            string_map = {s.address: s.text for s in strings}
            stage = time.perf_counter()
            decoder = Disassembler(image, self.max_instructions)
            candidate_instructions = decoder.run()
            recursive_seeds = {
                image.entry_point, *image.exports.keys(), *image.runtime_function_starts,
                *image.exception_handler_starts, *getattr(image, "go_functions", {}).keys(),
            }
            recursive_decode = getattr(decoder, "run_recursive", None)
            if callable(recursive_decode):
                instructions = recursive_decode(
                    recursive_seeds, candidate_instructions,
                    priority_seeds={image.entry_point},
                )
            else:
                # Backward compatibility for test doubles and external decoder adapters.
                instructions = candidate_instructions
                for ins in instructions:
                    ins.discovery_source = "legacy_decoder_adapter"
                    ins.boundary_confidence = "confirmed_reachable"
                decoder.warnings.append("Decoder adapter has no recursive pass; supplied instruction boundaries are trusted.")
            if not instructions:
                # Some malformed or unusual PEs have no usable entry metadata. Keep
                # analysis available while making the lower boundary confidence clear.
                instructions = candidate_instructions
                for ins in instructions:
                    ins.boundary_confidence = "linear_sweep_fallback"
                decoder.warnings.append(
                    "Recursive decoding found no reachable instructions; CFG uses linear-sweep candidates."
                )
            decode_seconds = time.perf_counter() - stage
            pointer_scan_start = time.perf_counter()
            static_pointer_slots, static_pointer_scan = image.scan_pointer_slots(
                max_slots=max(0, self.max_dataflow_instructions)
            )
            pointer_scan_seconds = time.perf_counter() - pointer_scan_start
            stage = time.perf_counter()
            cfg = ControlFlowGraph(
                instructions,
                extra_leaders={
                    image.entry_point, *image.exports.keys(), *image.runtime_function_starts,
                    *image.exception_handler_starts,
                    *(
                        target
                        for ins in instructions
                        for target in (ins.indirect_call_targets or ())
                    ),
                },
            )
            instr_by_addr = {x.address: x for x in instructions}
            def is_executable_address(address: int) -> bool:
                return any(
                    section.executable
                    and section.start <= address < min(
                        section.end,
                        section.start + getattr(section, "backed_size", section.end - section.start),
                    )
                    for section in image.sections
                )
            call_sites: dict[int, list[int]] = defaultdict(list)
            for ins in instructions:
                if ins.flow == "call" and ins.direct_target is not None:
                    call_sites[ins.direct_target].append(ins.address)
            function_reason: dict[int, set[str]] = defaultdict(set)
            def is_entry_candidate(addr: int) -> bool:
                ins = instr_by_addr.get(addr)
                return ins is not None and ins.mnemonic.lower() != ".byte"
            if is_entry_candidate(image.entry_point):
                function_reason[image.entry_point].add("PE entry point")
            for addr in image.exports:
                if is_entry_candidate(addr):
                    function_reason[addr].add("export")
            for addr in image.runtime_function_starts:
                if is_entry_candidate(addr):
                    function_reason[addr].add("Windows x64 runtime-function metadata")
            for addr in image.exception_handler_starts:
                if is_entry_candidate(addr):
                    function_reason[addr].add("Windows exception/unwind handler metadata")
            for target, sites in call_sites.items():
                if is_entry_candidate(target):
                    function_reason[target].add("direct call target")
            for ins in instructions:
                for target in ins.indirect_call_targets or ():
                    if is_entry_candidate(target):
                        function_reason[target].add("proven indirect call target")
            # A bounded indirect-jump table may be a function-pointer dispatcher
            # rather than a switch. Promote its destinations only when the case
            # vector is complete, has multiple distinct destinations, and every
            # destination begins with a narrow architecture-specific prologue.
            # Ordinary switch bodies and sparse/unbounded tables remain CFG blocks.
            for ins in instructions:
                slots = ins.jump_table_slots or []
                targets = sorted({target for target in slots if target is not None})
                if (ins.flow != "jump" or ins.direct_target is not None
                        or ins.jump_table_case_count is None or ins.jump_table_case_count < 2
                        or len(slots) != ins.jump_table_case_count
                        or any(target is None for target in slots) or len(targets) < 2):
                    continue
                if all(self._has_function_prologue(target, instr_by_addr, image.bits) for target in targets):
                    for target in targets:
                        if is_entry_candidate(target):
                            function_reason[target].add("complete jump-table function-dispatch candidate")
            # Recover isolated one-instruction jump thunks that forward directly
            # to an already established function entry. Restrict this to blocks
            # unreachable from all current function roots to avoid turning ordinary
            # in-function tail jumps into separate functions.
            known_entries = set(function_reason)
            rooted_blocks: set[int] = set()
            root_scan_truncated = False
            for known_entry in sorted(known_entries):
                reached = cfg.reachable_blocks(
                    known_entry, known_entries, self.max_blocks_per_function
                )
                rooted_blocks.update(reached)
                if len(reached) >= self.max_blocks_per_function:
                    root_scan_truncated = True
            thunk_candidates: set[int] = set()
            if not root_scan_truncated:
                for block_start, block in cfg.blocks.items():
                    if block_start in rooted_blocks or len(block.instructions) != 1:
                        continue
                    thunk = block.instructions[0]
                    if (thunk.mnemonic.lower() in {"jmp", "jmpq"}
                            and thunk.direct_target in known_entries
                            and is_entry_candidate(block_start)):
                        function_reason[block_start].add("isolated direct-jump thunk to known function")
                        thunk_candidates.add(block_start)
            # Retain only candidates that are actual decoded instruction starts.
            candidates = sorted(function_reason)
            def enforce_function_cap(addresses: list[int]) -> list[int]:
                if len(addresses) <= self.max_functions:
                    return addresses
                # Keep the PE application root even when a low-address symbol
                # population would otherwise consume the entire candidate budget.
                mandatory = [image.entry_point] if image.entry_point in addresses else []
                retained = mandatory + [address for address in addresses if address not in mandatory]
                return sorted(retained[:self.max_functions])
            if len(candidates) > self.max_functions:
                candidates = enforce_function_cap(candidates)
                function_cap_warning = True
            else:
                function_cap_warning = False
            pseudocode_candidates = candidates[:self.max_pseudocode_functions]
            names: dict[int, str] = {}
            for addr in candidates:
                if addr in image.exports:
                    names[addr] = image.exports[addr]
                elif addr in getattr(image, "go_functions", {}):
                    names[addr] = image.go_functions[addr]
                else:
                    names[addr] = f"sub_{addr:X}"
            # Turn tiny IAT jump stubs into their imported function names when evidence is direct.
            for addr in candidates:
                ins = instr_by_addr.get(addr)
                if ins and ins.flow == "jump" and ins.reference_target in image.imports:
                    names[addr] = image.imports[ins.reference_target].split("!")[-1]
            function_entry_set = set(candidates)
            functions: list[FunctionInfo] = []
            function_blocks: dict[int, list[Any]] = {}
            xref_rows: list[dict[str, str]] = []
            for ins in instructions:
                if ins.reference_target is not None:
                    if ins.reference_target in string_map:
                        xref_rows.append({"source": hex(ins.address), "target": hex(ins.reference_target), "kind": "string", "value": string_map[ins.reference_target]})
                    if ins.reference_target in image.imports:
                        xref_rows.append({"source": hex(ins.address), "target": hex(ins.reference_target), "kind": "import", "value": image.imports[ins.reference_target]})
            string_rows_by_address = {s.address: s.to_dict() for s in strings}
            for candidate_index, entry in enumerate(candidates):
                if not self._progress("function_index", candidate_index, len(candidates)):
                    break
                block_ids = cfg.reachable_blocks(entry, function_entry_set, self.max_blocks_per_function)
                blks = [cfg.blocks[b] for b in block_ids if b in cfg.blocks]
                if entry in pseudocode_candidates:
                    function_blocks[entry] = blks
                f_ins = [ins for b in blks for ins in b.instructions]
                f_calls = sorted({
                    target
                    for ins in f_ins
                    if ins.flow == "call" or (ins.flow == "jump" and ins.direct_target in function_entry_set)
                    for target in ((ins.direct_target,) if ins.direct_target is not None else tuple(ins.indirect_call_targets or ()))
                })
                f_strings = []
                seen_strs = set()
                for ins in f_ins:
                    if ins.reference_target in string_rows_by_address and ins.reference_target not in seen_strs:
                        f_strings.append(string_rows_by_address[ins.reference_target]); seen_strs.add(ins.reference_target)
                functions.append(FunctionInfo(entry, names[entry], ", ".join(sorted(function_reason[entry])), block_ids,
                                              f_calls, f_strings, len(f_ins)))
            if not self.cancelled:
                self._progress("function_index", len(candidates), len(candidates))
            else:
                candidates = [function.address for function in functions]
                pseudocode_candidates = [entry for entry in pseudocode_candidates if entry in candidates]
                function_entry_set = set(candidates)
                names = {entry: names[entry] for entry in candidates}

            # A bounded discovery pass can promote indirect calls and unscaled
            # indirect tail jumps whose values resolve to decoded code addresses.
            # It has its own row cap equal to the data-flow budget; regular report
            # output keeps a separate cap and records the discovery cost below.
            indirect_call_discovery = {
                "enabled": self.write_dataflow, "row_budget": self.max_dataflow_instructions,
                "function_budget": self.max_dataflow_functions,
                "functions_available": len(pseudocode_candidates),
                "rows_scanned": 0, "functions_scanned": 0, "budget_exhausted": False,
                "function_budget_exhausted": self.write_dataflow and len(pseudocode_candidates) > self.max_dataflow_functions,
                "targets_resolved": [], "tail_jump_targets": [], "callback_targets": [], "new_function_candidates": [],
                "newly_decoded_targets": [], "rejected_non_executable_targets": [],
            }
            discovered_targets: set[int] = set()
            if not self.cancelled and self.write_dataflow and function_blocks and self.max_dataflow_instructions > 0:
                discovery_flow = FunctionDataFlow(
                    image.bits, string_map, image.imports, names,
                    known_code_addresses=set(instr_by_addr) | {ins.address for ins in candidate_instructions},
                    max_steps_per_function=max(2_000, min(20_000, self.max_dataflow_instructions)),
                    static_pointer_slots=static_pointer_slots,
                )
                discovery_budget = self.max_dataflow_instructions
                resolved_targets: set[int] = set()
                tail_jump_targets: set[int] = set()
                callback_targets: set[int] = set()
                pending_decode_targets: set[int] = set()
                rejected_non_executable_targets: set[int] = set()
                pending_call_edges: list[tuple[int, int, str | None]] = []
                pending_jump_edges: list[tuple[int, int]] = []
                pending_callbacks: set[int] = set()
                discovery_candidates = pseudocode_candidates[:self.max_dataflow_functions]
                for source_index, source_entry in enumerate(discovery_candidates):
                    if not self._progress("indirect_call_discovery", source_index, len(discovery_candidates)):
                        break
                    if discovery_budget <= 0:
                        indirect_call_discovery["budget_exhausted"] = source_index < len(discovery_candidates)
                        break
                    source_blocks = function_blocks.get(source_entry, [])
                    rows, _ = discovery_flow.analyze(source_entry, source_blocks, max_rows=discovery_budget)
                    indirect_call_discovery["functions_scanned"] += 1
                    indirect_call_discovery["rows_scanned"] += len(rows)
                    discovery_budget -= len(rows)
                    for row in rows:
                        try:
                            call_address = int(str(row["address"]), 0)
                        except (KeyError, TypeError, ValueError):
                            call_address = None
                        call_ins = instr_by_addr.get(call_address) if call_address is not None else None
                        target_text = row.get("resolved_call_target")
                        if target_text:
                            try:
                                target = int(str(target_text), 0)
                            except ValueError:
                                target = None
                            if (target is not None and call_ins is not None and call_ins.flow == "call"
                                    and call_ins.direct_target is None):
                                if not is_executable_address(target):
                                    rejected_non_executable_targets.add(target)
                                elif (target not in instr_by_addr or
                                      instr_by_addr[target].boundary_confidence != "confirmed_reachable"):
                                    pending_decode_targets.add(target)
                                    pending_call_edges.append((call_ins.address, target, row.get("resolved_call_provenance")))
                                else:
                                    call_ins.indirect_call_targets = sorted(set(call_ins.indirect_call_targets or ()) | {target})
                                    call_ins.resolved_call_provenance = row.get("resolved_call_provenance") or "value_flow"
                                    call_ins.unresolved_call_reason = None
                                    resolved_targets.add(target)
                                    function_reason[target].add("resolved indirect call target (value flow)")
                                    discovered_targets.add(target)
                        elif (call_ins is not None and call_ins.flow == "call"
                              and call_ins.direct_target is None and not call_ins.import_name
                              and not call_ins.indirect_call_targets):
                            call_ins.unresolved_call_reason = (
                                row.get("resolved_call_reason") or
                                "value-flow analysis found no concrete call target"
                            )
                        jump_target_text = row.get("resolved_indirect_jump_target")
                        if jump_target_text:
                            try:
                                target = int(str(jump_target_text), 0)
                            except ValueError:
                                target = None
                            if (target is not None and call_ins is not None and call_ins.flow == "jump"
                                    and call_ins.direct_target is None and call_ins.jump_table_case_count is None):
                                if not is_executable_address(target):
                                    rejected_non_executable_targets.add(target)
                                elif (target not in instr_by_addr or
                                      instr_by_addr[target].boundary_confidence != "confirmed_reachable"):
                                    pending_decode_targets.add(target)
                                    pending_jump_edges.append((call_ins.address, target))
                                else:
                                    call_ins.resolved_indirect_jump_target = target
                                    call_ins.indirect_targets = sorted(set(call_ins.indirect_targets or ()) | {target})
                                    tail_jump_targets.add(target)
                                    function_reason[target].add("resolved indirect tail-jump target (value flow)")
                                    discovered_targets.add(target)
                        # A decoded code address passed in a Windows x64 register
                        # argument is evidence of a callback/function-pointer target.
                        for argument in (
                            list(row.get("call_argument_values") or ())
                            + list(row.get("call_stack_memory_values") or ())
                        ):
                            try:
                                callback = int(str(argument), 0)
                            except (TypeError, ValueError):
                                continue
                            if not is_executable_address(callback):
                                rejected_non_executable_targets.add(callback)
                                continue
                            if (callback not in instr_by_addr or
                                    instr_by_addr[callback].boundary_confidence != "confirmed_reachable"):
                                pending_decode_targets.add(callback)
                                pending_callbacks.add(callback)
                                continue
                            callback_targets.add(callback)
                            function_reason[callback].add("code pointer passed as call argument (callback candidate)")
                            discovered_targets.add(callback)
                    if discovery_budget <= 0 and source_index + 1 < len(discovery_candidates):
                        indirect_call_discovery["budget_exhausted"] = True
                if pending_decode_targets and not self.cancelled:
                    newly_decoded = decoder.run_recursive(
                        pending_decode_targets, candidate_instructions, instructions
                    )
                    old_addresses = set(instr_by_addr)
                    instructions = newly_decoded
                    instr_by_addr = {ins.address: ins for ins in instructions}
                    indirect_call_discovery["newly_decoded_targets"] = [
                        hex(address) for address in sorted(pending_decode_targets)
                        if address in instr_by_addr and address not in old_addresses
                    ]
                    for source, target, provenance in pending_call_edges:
                        target_ins = instr_by_addr.get(target)
                        call_ins = instr_by_addr.get(source)
                        if target_ins is None or target_ins.boundary_confidence != "confirmed_reachable" or call_ins is None:
                            continue
                        call_ins.indirect_call_targets = sorted(set(call_ins.indirect_call_targets or ()) | {target})
                        call_ins.resolved_call_provenance = provenance or "value_flow"
                        call_ins.unresolved_call_reason = None
                        resolved_targets.add(target)
                        function_reason[target].add("resolved indirect call target (value flow)")
                        discovered_targets.add(target)
                    for source, target in pending_jump_edges:
                        target_ins = instr_by_addr.get(target)
                        call_ins = instr_by_addr.get(source)
                        if target_ins is None or target_ins.boundary_confidence != "confirmed_reachable" or call_ins is None:
                            continue
                        call_ins.resolved_indirect_jump_target = target
                        call_ins.indirect_targets = sorted(set(call_ins.indirect_targets or ()) | {target})
                        tail_jump_targets.add(target)
                        function_reason[target].add("resolved indirect tail-jump target (value flow)")
                        discovered_targets.add(target)
                    for callback in pending_callbacks:
                        target_ins = instr_by_addr.get(callback)
                        if target_ins is None or target_ins.boundary_confidence != "confirmed_reachable":
                            continue
                        callback_targets.add(callback)
                        function_reason[callback].add("code pointer passed as call argument (callback candidate)")
                        discovered_targets.add(callback)
                if not self.cancelled:
                    self._progress("indirect_call_discovery", len(discovery_candidates), len(discovery_candidates))
                indirect_call_discovery["targets_resolved"] = [hex(x) for x in sorted(resolved_targets)]
                indirect_call_discovery["tail_jump_targets"] = [hex(x) for x in sorted(tail_jump_targets)]
                indirect_call_discovery["callback_targets"] = [hex(x) for x in sorted(callback_targets)]
                indirect_call_discovery["new_function_candidates"] = [hex(x) for x in sorted(discovered_targets)]
                indirect_call_discovery["rejected_non_executable_targets"] = [
                    hex(x) for x in sorted(rejected_non_executable_targets)
                ]

                if (resolved_targets or tail_jump_targets or callback_targets) and not self.cancelled:
                    # Rebuild block boundaries with the recovered entries so a
                    # target cannot remain embedded in another function's block.
                    cfg = ControlFlowGraph(
                        instructions,
                        extra_leaders={
                            image.entry_point, *image.exports.keys(), *image.runtime_function_starts,
                            *image.exception_handler_starts,
                            *(target for ins in instructions for target in (ins.indirect_targets or ())),
                            *(target for ins in instructions for target in (ins.indirect_call_targets or ())),
                        },
                    )
                    candidates = sorted(function_reason)
                    if len(candidates) > self.max_functions:
                        candidates = enforce_function_cap(candidates)
                        function_cap_warning = True
                    pseudocode_candidates = candidates[:self.max_pseudocode_functions]
                    names = {addr: image.exports.get(addr, f"sub_{addr:X}") for addr in candidates}
                    for addr in candidates:
                        stub = instr_by_addr.get(addr)
                        if stub and stub.flow == "jump" and stub.reference_target in image.imports:
                            names[addr] = image.imports[stub.reference_target].split("!")[-1]
                    function_entry_set = set(candidates)
                    functions = []
                    function_blocks = {}
                    for candidate_index, entry in enumerate(candidates):
                        if not self._progress("function_rebuild", candidate_index, len(candidates)):
                            break
                        block_ids = cfg.reachable_blocks(entry, function_entry_set, self.max_blocks_per_function)
                        blks = [cfg.blocks[b] for b in block_ids if b in cfg.blocks]
                        if entry in pseudocode_candidates:
                            function_blocks[entry] = blks
                        f_ins = [ins for b in blks for ins in b.instructions]
                        f_calls = sorted({
                            target for ins in f_ins
                            if (ins.flow == "call"
                                    or (ins.flow == "jump" and (
                                        ins.direct_target in function_entry_set
                                        or ins.resolved_indirect_jump_target in function_entry_set
                                    )))
                            for target in (
                                (ins.direct_target,) if ins.direct_target is not None else
                                tuple(ins.indirect_call_targets or ()) +
                                ((ins.resolved_indirect_jump_target,) if ins.resolved_indirect_jump_target is not None else ())
                            )
                        })
                        f_strings = []
                        seen_strs = set()
                        for ins in f_ins:
                            if ins.reference_target in string_rows_by_address and ins.reference_target not in seen_strs:
                                f_strings.append(string_rows_by_address[ins.reference_target])
                                seen_strs.add(ins.reference_target)
                        functions.append(FunctionInfo(
                            entry, names[entry], ", ".join(sorted(function_reason[entry])),
                            block_ids, f_calls, f_strings, len(f_ins),
                        ))
                    if self.cancelled:
                        candidates = [function.address for function in functions]
                        pseudocode_candidates = [entry for entry in pseudocode_candidates if entry in candidates]
                        function_entry_set = set(candidates)
                        names = {entry: names[entry] for entry in candidates}
                    else:
                        self._progress("function_rebuild", len(candidates), len(candidates))
            cfg_seconds = time.perf_counter() - stage
            # Positive application reachability is computed from the PE entry root
            # over recovered function call edges. A negative result is unknown
            # because indirect edges and function boundaries can be incomplete.
            function_by_address = {function.address: function for function in functions}
            application_reachable: set[int] = set()
            if image.entry_point in function_by_address:
                application_reachable.add(image.entry_point)
                reach_pending = [image.entry_point]
                while reach_pending:
                    current = reach_pending.pop()
                    for target in function_by_address[current].calls:
                        if target in function_by_address and target not in application_reachable:
                            application_reachable.add(target)
                            reach_pending.append(target)
            function_analysis_states: dict[int, dict[str, Any]] = {}
            go_symbol_addresses = set(getattr(image, "go_functions", {}))
            for address in function_by_address:
                is_symbol = address in image.exports or address in go_symbol_addresses
                is_seed = address in recursive_seeds or address in discovered_targets
                reachable_state = True if address in application_reachable else None
                states = []
                if is_seed:
                    states.append("decoded_from_seed")
                if reachable_state is True:
                    states.append("reachable_from_application_root")
                if is_symbol:
                    states.append("symbol_present")
                if reachable_state is None:
                    states.append("unknown")
                function_analysis_states[address] = {
                    "states": states,
                    "decoded_from_seed": is_seed,
                    "reachable_from_application_root": reachable_state,
                    "symbol_present": is_symbol,
                }
            stage = time.perf_counter()
            # Resolved import target is attached to references as well as the original operand string.
            for ins in instructions:
                if ins.import_name is None and ins.reference_target in image.imports:
                    ins.import_name = image.imports[ins.reference_target]
            crypto_report = analyze_crypto(
                strings, image.imports, instructions=instructions, artifact_path=self.input_path,
                go_functions=getattr(image, "go_functions", {}),
                go_symbol_metadata=getattr(image, "go_symbol_metadata", {}),
                go_build_metadata=getattr(image, "go_build_metadata", {}),
                executable_ranges=[(sec.start, sec.start + sec.backed_size)
                                   for sec in image.sections if sec.executable],
                application_reachable_functions=application_reachable,
            )
            lifter = PseudocodeLifter(image.bits, string_map, image.imports, names)
            # Recover ABI register-argument counts and EAX/RAX return behavior before
            # lifting calls/returns, so callsites can use these signatures.
            signature_rows = lifter.infer_function_signatures(function_blocks)
            self._json_write(self.output_dir / "function_signatures.json", signature_rows)

            # General data-flow pass: bounded forward propagation through basic blocks.
            # This is a foundation for later SSA/variable reconstruction; it intentionally
            # does not claim to recover original source names or exact C syntax.
            dataflow_seconds = 0.0
            ir_seconds = 0.0
            ssa_seconds = 0.0
            normalized_ir_count = 0
            ssa_operation_count = 0
            ssa_phi_count = 0
            ssa_function_count = 0
            ssa_function_summaries: list[dict[str, Any]] = []
            ssa_warnings: list[str] = []
            dataflow_rows_count = 0
            dataflow_functions_count = 0
            dataflow_warnings: list[str] = []
            dataflow_summary_rows: list[dict[str, Any]] = []
            dataflow_conditions: dict[int, dict[int, str]] = defaultdict(dict)
            if self.write_dataflow:
                df_start = time.perf_counter()
                arities = {entry: self._infer_function_arity(function_blocks.get(entry, []), image.bits)
                           for entry in pseudocode_candidates}
                df = FunctionDataFlow(
                    image.bits, string_map, image.imports, names,
                    function_arg_counts={k: v for k, v in arities.items() if v is not None},
                    function_return_types={
                        address: str(signature.get("return_type"))
                        for address, signature in lifter.function_signatures.items()
                        if signature.get("return_type") in {
                            "uint32_t", "uint64_t", "uintptr_t", "size_t",
                            "bool", "int32_t", "void",
                        }
                    },
                    max_steps_per_function=max(2_000, min(20_000, self.max_dataflow_instructions)),
                    static_pointer_slots=static_pointer_slots,
                )
                value_flow_path = self.output_dir / "value_flow.jsonl"
                dataflow_budget = self.max_dataflow_instructions
                with value_flow_path.open("w", encoding="utf-8") as vf:
                    dataflow_candidates = pseudocode_candidates[:self.max_dataflow_functions]
                    for function_index, entry in enumerate(dataflow_candidates):
                        if not self._progress("dataflow", function_index, len(dataflow_candidates)):
                            break
                        if dataflow_budget <= 0:
                            dataflow_warnings.append(f"Global data-flow output budget reached ({self.max_dataflow_instructions:,} rows).")
                            break
                        blocks_for_fn = function_blocks.get(entry, [])
                        rows, df_summary = df.analyze(entry, blocks_for_fn, max_rows=dataflow_budget)
                        for row in rows:
                            condition = row.get("condition")
                            if isinstance(condition, str):
                                try:
                                    condition_address = int(str(row.get("address")), 0)
                                except (TypeError, ValueError):
                                    condition_address = None
                                if condition_address is not None:
                                    dataflow_conditions[entry][condition_address] = condition
                            resolved_import = row.get("resolved_call_import")
                            if resolved_import:
                                try:
                                    call_address = int(str(row.get("address")), 0)
                                except (TypeError, ValueError):
                                    call_address = None
                                call_ins = instr_by_addr.get(call_address) if call_address is not None else None
                                if call_ins is not None and call_ins.flow == "call" and call_ins.direct_target is None:
                                    call_ins.import_name = str(resolved_import)
                            vf.write(json.dumps(row, ensure_ascii=False) + "\n")
                        dataflow_rows_count += len(rows)
                        dataflow_functions_count += 1
                        dataflow_budget -= len(rows)
                        dataflow_summary_rows.append(df_summary)
                        dataflow_warnings.extend(df_summary.get("warnings", []))
                    if not self.cancelled:
                        self._progress("dataflow", len(dataflow_candidates), len(dataflow_candidates))
                    else:
                        dataflow_warnings.append("Analysis cancelled during data-flow; reports contain completed functions only.")
                dataflow_warnings = list(dict.fromkeys(dataflow_warnings))
                self._json_write(self.output_dir / "dataflow_summary.json", {
                    "functions_analyzed": dataflow_functions_count,
                    "value_flow_rows": dataflow_rows_count,
                    "row_budget": self.max_dataflow_instructions,
                    "function_budget": self.max_dataflow_functions,
                    "per_function": dataflow_summary_rows,
                    "warnings": list(dict.fromkeys(dataflow_warnings)),
                })
                dataflow_seconds = time.perf_counter() - df_start

            # New independent IR + SSA stage. It uses only our own normalized operations,
            # CFG and dominator analysis; it never launches or imports Ghidra.
            if self.write_ssa:
                ir_start = time.perf_counter()
                normalized_path = self.output_dir / "normalized_ir.jsonl"
                ssa_path = self.output_dir / "ssa_ir.jsonl"
                ssa_cfg_rows: list[dict[str, Any]] = []
                ssa_budget = self.max_dataflow_instructions
                with normalized_path.open("w", encoding="utf-8") as nir, ssa_path.open("w", encoding="utf-8") as sir:
                    ssa_candidates = pseudocode_candidates[:self.max_dataflow_functions]
                    for function_index, entry in enumerate(ssa_candidates):
                        if not self._progress("normalized_ir_ssa", function_index, len(ssa_candidates)):
                            break
                        blocks_for_fn = function_blocks.get(entry, [])
                        if not blocks_for_fn:
                            continue
                        ir_rows = lift_blocks(blocks_for_fn, image.bits)
                        if len(ir_rows) > ssa_budget:
                            ssa_warnings.append(
                                f"SSA output budget reached ({self.max_dataflow_instructions:,} operations); remaining functions skipped to avoid partial-function SSA."
                            )
                            break
                        for row in ir_rows:
                            row["function_entry"] = hex(entry)
                            row["function_name"] = names.get(entry, f"sub_{entry:X}")
                            nir.write(json.dumps(row, ensure_ascii=False) + "\n")
                        normalized_ir_count += len(ir_rows)
                        ssa_result = construct_ssa(blocks_for_fn, ir_rows, entry)
                        fn_ssa_ops = ssa_result.get("operations", [])
                        for row in fn_ssa_ops:
                            row["function_entry"] = hex(entry)
                            row["function_name"] = names.get(entry, f"sub_{entry:X}")
                            sir.write(json.dumps(row, ensure_ascii=False) + "\n")
                        for block_row in ssa_result.get("blocks", []):
                            block_row["function_entry"] = hex(entry)
                            block_row["function_name"] = names.get(entry, f"sub_{entry:X}")
                            ssa_cfg_rows.append(block_row)
                        ss = dict(ssa_result.get("summary", {}))
                        ss["function_entry"] = hex(entry)
                        ss["function_name"] = names.get(entry, f"sub_{entry:X}")
                        ssa_function_summaries.append(ss)
                        ssa_operation_count += len(fn_ssa_ops)
                        ssa_phi_count += int(ss.get("phi_count", 0))
                        ssa_function_count += 1
                        ssa_budget -= len(ir_rows)
                    if not self.cancelled:
                        self._progress("normalized_ir_ssa", len(ssa_candidates), len(ssa_candidates))
                ir_seconds = time.perf_counter() - ir_start
                ssa_seconds = ir_seconds
                self._json_write(self.output_dir / "ssa_cfg.json", ssa_cfg_rows)
                self._json_write(self.output_dir / "ssa_summary.json", {
                    "engine": "independent-normalized-ir-and-ssa-v1",
                    "ghidra_used": False,
                    "functions_analyzed": ssa_function_count,
                    "normalized_ir_operations": normalized_ir_count,
                    "ssa_operations": ssa_operation_count,
                    "phi_nodes": ssa_phi_count,
                    "operation_budget": self.max_dataflow_instructions,
                    "function_budget": self.max_dataflow_functions,
                    "per_function": ssa_function_summaries,
                    "capabilities": [
                        "instruction normalization for common x86/x64 operations",
                        "iterative dominators and immediate dominators",
                        "dominance frontiers",
                        "iterated-dominance-frontier phi placement",
                        "SSA versioning for general registers and simple RBP/RSP-relative stack slots",
                    ],
                    "limitations": [
                        "not complete x86 semantics; unsupported instructions remain opaque",
                        "SSA models registers and stack slots; exact non-stack memory facts are tracked only by the bounded data-flow pass",
                        "register and stack variable analysis is heuristic and not type recovery",
                        "function boundaries inherit the existing heuristic CFG recovery",
                    ],
                    "warnings": list(dict.fromkeys(ssa_warnings)),
                })
            stage = time.perf_counter()
            pseudocode_path = self.output_dir / "pseudocode.c.txt"
            pseudocode_emitted_functions = 0
            pseudocode_emitted_instructions = 0
            pseudocode_truncated = False
            decompiler_rows: list[dict[str, Any]] = []
            with pseudocode_path.open("w", encoding="utf-8") as f:
                f.write("/* Independent static decompiler research engine. Output is heuristic pseudocode, not recovered original source. */\n")
                f.write(f"/* Input: {Path(self.input_path).name}; architecture: {image.architecture}; decoded instructions: {len(instructions)} */\n\n")
                for function_index, entry in enumerate(pseudocode_candidates):
                    if not self._progress("pseudocode", function_index, len(pseudocode_candidates)):
                        break
                    if pseudocode_emitted_instructions >= self.max_pseudocode_instructions:
                        pseudocode_truncated = True
                        break
                    original_blocks = function_blocks.get(entry, [])
                    emitted_blocks = []
                    for block in original_blocks:
                        remaining = self.max_pseudocode_instructions - pseudocode_emitted_instructions - sum(len(b.instructions) for b in emitted_blocks)
                        if remaining <= 0:
                            pseudocode_truncated = True
                            break
                        if len(block.instructions) <= remaining:
                            emitted_blocks.append(block)
                        else:
                            from .models import BasicBlock
                            emitted_blocks.append(BasicBlock(block.start, block.end, block.instructions[:remaining], [], block.section))
                            pseudocode_truncated = True
                            break
                    emitted_count = sum(len(b.instructions) for b in emitted_blocks)
                    function_text = lifter.function_text(
                        names[entry], entry, emitted_blocks,
                        dataflow_conditions.get(entry, {}),
                    )
                    f.write(function_text)
                    f.write("\n")
                    signature = next((row for row in signature_rows if row.get("address") == hex(entry)), None)
                    decompiler_rows.append({
                        "address": hex(entry),
                        "name": names[entry],
                        "signature": signature,
                        "blocks": [block.to_dict() for block in emitted_blocks],
                        "code": function_text,
                        "analysis": {
                            "instruction_count": emitted_count,
                            "complete": not pseudocode_truncated,
                            "source": "heuristic-register-and-stack-lifter",
                        },
                    })
                    pseudocode_emitted_instructions += emitted_count
                    pseudocode_emitted_functions += 1
                    if pseudocode_truncated:
                        break
                if not self.cancelled:
                    self._progress("pseudocode", pseudocode_emitted_functions, len(pseudocode_candidates))
            export_warnings = list(decoder.warnings)
            if self.cancelled:
                export_warnings.append(
                    f"Analysis cancelled during {self.cancelled_stage or 'analysis'}; generated reports contain completed work and are partial."
                )
            if function_cap_warning:
                export_warnings.append(f"Function cap reached ({self.max_functions:,}); function output is partial.")
            if len(candidates) > len(pseudocode_candidates):
                export_warnings.append(f"Pseudocode function cap reached ({self.max_pseudocode_functions:,}); functions.json remains broader than pseudocode.c.txt.")
            if pseudocode_truncated:
                export_warnings.append(f"Pseudocode instruction budget reached ({self.max_pseudocode_instructions:,}); pseudocode.c.txt is truncated to keep large outputs manageable.")
            if any(len(f.block_starts) >= self.max_blocks_per_function for f in functions):
                export_warnings.append(f"At least one function reached the per-function block cap ({self.max_blocks_per_function:,}); some function CFGs may be truncated.")
            export_warnings += dataflow_warnings
            export_warnings += [
                "The CFG uses recursively reached instruction boundaries. Linear-sweep candidates remain available in assembly output and may include embedded data; unreachable code can be omitted.",
                "Function starts are heuristic: PE entry, exports, runtime metadata, direct calls, and bounded value-flow-resolved indirect calls; unresolved indirect calls and computed jumps may be missed.",
                "Pseudocode is conservative; structured source control flow, full type recovery and original source names are not generally recovered.",
                "Return types and call arguments are heuristic; full ABI typing and arbitrary memory-held function pointers are not yet recovered.",
                "Indirect-call discovery is bounded by the value-flow row/function budgets and requires a concrete target matching a decoded instruction start.",
                "Windows x64 unwind operations and handler references are reported; exceptional control-flow edges are not modeled."
            ]
            export_warnings += ssa_warnings
            indirect_control_transfers = []
            for ins in instructions:
                is_candidate_jump_table = (
                    ins.flow == "jump" and ins.direct_target is None and ins.reference_target is not None
                )
                is_resolved_indirect_jump = (
                    ins.flow == "jump" and ins.resolved_indirect_jump_target is not None
                )
                is_resolved_indirect_call = ins.flow == "call" and (
                    bool(ins.indirect_call_targets)
                    or (ins.direct_target is None and bool(ins.import_name))
                )
                if not (is_candidate_jump_table or is_resolved_indirect_jump or is_resolved_indirect_call):
                    continue
                indirect_control_transfers.append({
                    "source": hex(ins.address),
                    "kind": (
                        "indirect_call" if ins.flow == "call" else
                        "indirect_tail_jump" if is_resolved_indirect_jump else "indirect_jump"
                    ),
                    "reference": hex(ins.reference_target) if ins.reference_target is not None else None,
                    "import": ins.import_name,
                    "targets": [hex(x) for x in (
                        ins.indirect_call_targets if ins.flow == "call" else ins.indirect_targets
                    ) or []],
                    "jump_table": ({
                        "base": hex(ins.jump_table_base) if ins.jump_table_base is not None else None,
                        "entry_size": ins.jump_table_entry_size,
                        "entries_observed": ins.jump_table_entries_observed,
                        "index_register": ins.jump_table_index_register,
                        "index_bias": ins.jump_table_index_bias,
                        "case_count_from_bound_check": ins.jump_table_case_count,
                        "slots": [hex(target) if target is not None else None for target in (ins.jump_table_slots or [])],
                        "default_target": hex(ins.jump_table_default_target) if ins.jump_table_default_target is not None else None,
                        "default_entry_indices": ins.jump_table_default_entry_indices,
                        "bound_branch": ins.jump_table_bound_branch,
                        "bound_instruction": hex(ins.jump_table_bound_instruction) if ins.jump_table_bound_instruction is not None else None,
                    } if is_candidate_jump_table else None),
                    "resolved_target": (
                        hex(ins.resolved_indirect_jump_target)
                        if ins.resolved_indirect_jump_target is not None else None
                    ),
                })
            unwind_report = []
            for record in image.runtime_functions:
                chained = None
                if record.get("chained_start") is not None:
                    chained = {
                        "start": hex(record["chained_start"]),
                        "end": hex(record["chained_end"]),
                        "unwind_info": hex(record["chained_unwind_info"]),
                    }
                handler_va = record.get("handler_va")
                language_data_va = record.get("language_data_va")
                unwind_report.append({
                    "start": hex(record["start"]),
                    "end": hex(record["end"]),
                    "unwind_info": hex(record["unwind_info"]),
                    "version": record.get("unwind_version"),
                    "flags": record.get("unwind_flags"),
                    "flag_names": record.get("unwind_flag_names", []),
                    "prolog_size": record.get("prolog_size"),
                    "code_slots_declared": record.get("unwind_code_count"),
                    "frame_register": record.get("frame_register_name"),
                    "frame_register_scale": record.get("frame_register_scale"),
                    "operations": record.get("unwind_operations", []),
                    "handler_rva": record.get("handler_rva"),
                    "handler": hex(handler_va) if isinstance(handler_va, int) else None,
                    "language_data_offset": record.get("language_data_offset"),
                    "language_data": hex(language_data_va)
                    if isinstance(language_data_va, int) else None,
                    "chained_function": chained,
                    "warnings": record.get("unwind_parse_warnings", []),
                })
            # Keep exception/unwind relationships separate from ordinary CFG
            # successors: UNWIND_INFO identifies a handler associated with a
            # protected runtime-function range, but does not describe when that
            # handler is reached or the language-specific scope semantics.
            exceptional_edges = []
            chained_unwind_relationships = []
            recovered_function_entries = {function.address for function in functions}
            for record in image.runtime_functions:
                chained_start = record.get("chained_start")
                chained_end = record.get("chained_end")
                source_start = record.get("start")
                if (isinstance(source_start, int) and isinstance(chained_start, int)
                        and isinstance(chained_end, int) and chained_start < chained_end):
                    chained_unwind_relationships.append({
                        "kind": "windows_x64_chained_runtime_function",
                        "source_function": hex(source_start),
                        "target_function": hex(chained_start),
                        "target_range": {"start": hex(chained_start), "end": hex(chained_end)},
                        "target_function_recovered": chained_start in recovered_function_entries,
                        "unwind_info": hex(record["chained_unwind_info"])
                        if isinstance(record.get("chained_unwind_info"), int) else None,
                    })
                handler_va = record.get("handler_va")
                if not isinstance(handler_va, int) or handler_va <= 0:
                    continue
                start = record.get("start")
                end = record.get("end")
                if not isinstance(start, int) or not isinstance(end, int) or start >= end:
                    continue
                exceptional_edges.append({
                    "kind": "windows_x64_unwind_handler_candidate",
                    "source_function": hex(start),
                    "protected_range": {"start": hex(start), "end": hex(end)},
                    "target_handler": hex(handler_va),
                    "target_function_recovered": handler_va in recovered_function_entries,
                    "unwind_info": hex(record["unwind_info"])
                    if isinstance(record.get("unwind_info"), int) else None,
                    "flags": record.get("unwind_flag_names", []),
                    "control_flow_proven": False,
                })
            # These reports intentionally use a documented, Ghidra-inspired schema
            # rather than copying Ghidra's internal database or proprietary formats.
            # They expose the same useful concepts: program metadata, memory blocks,
            # symbols, functions, references, and decompiler text.
            program_report = {
                "format": "independent-ghidra-like-v1",
                "tool": {
                    "name": "Independent Decompiler Engine",
                    "version": "0.4.0-dev",
                    "ghidra_used": False,
                },
                "program": {
                    "name": Path(self.input_path).name,
                    "path": self.input_path,
                    "format": "Portable Executable",
                    "language": f"x86:{'LE:64' if image.bits == 64 else 'LE:32'}",
                    "image_base": hex(image.image_base),
                    "entry_point": hex(image.entry_point),
                },
                "memory_blocks": [
                    {
                        "name": section.name,
                        "start": hex(section.start),
                        "end": hex(section.end),
                        "size": section.end - section.start,
                        "initialized_size": section.raw_size,
                        "permissions": "".join(
                            flag for flag, enabled in (
                                ("r", section.readable),
                                ("w", section.writable),
                                ("x", section.executable),
                            ) if enabled
                        ),
                    }
                    for section in image.sections
                ],
                "symbols": {
                    "imports": [
                        {"address": hex(address), "name": symbol, "external": True}
                        for address, symbol in sorted(image.imports.items())
                    ],
                    "exports": [
                        {"address": hex(address), "name": name, "external": False}
                        for address, name in sorted(image.exports.items())
                    ],
                },
                "functions": [
                    {
                        "address": hex(function.address),
                        "name": function.name,
                        "namespace": "global",
                        "discovery": function.reason,
                        "blocks": [hex(block) for block in function.block_starts],
                        "calls": [hex(call) for call in function.calls],
                        "analysis_states": function_analysis_states.get(function.address, {"states": ["unknown"]}),
                        "strings": function.strings,
                        "instruction_count": function.instruction_count,
                        "signature": next(
                            (row for row in signature_rows if row.get("address") == hex(function.address)),
                            None,
                        ),
                    }
                    for function in functions
                ],
                "references": xref_rows,
                "indirect_control_transfers": indirect_control_transfers,
                "runtime_function_unwind": unwind_report,
                "chained_unwind_relationships": chained_unwind_relationships,
                "exceptional_edges": exceptional_edges,
                "analysis": {
                    "instruction_count": len(instructions),
                    "code_discovery": {
                        "method": "recursive_control_flow_with_linear_sweep_candidates",
                        "recursive_seed_count": len(recursive_seeds),
                        "recursive_instruction_count": sum(
                            ins.discovery_source == "recursive_control_flow" for ins in instructions
                        ),
                        "linear_sweep_candidate_instruction_count": len(candidate_instructions),
                    },
                    "basic_block_count": len(cfg.blocks),
                    "string_count": len(strings),
                    "crypto_indicator_summary": {
                        "analyzer": crypto_report["analyzer"],
                        "algorithm_names_found": crypto_report["algorithm_names_found"],
                        "algorithm_use_states": crypto_report["algorithm_use_states"],
                        "algorithm_families": crypto_report["algorithm_families"],
                        "go_function_symbol_names": crypto_report["algorithm_evidence_tiers"]["go_function_symbol_names"],
                        "algorithm_string_hits": crypto_report["algorithm_string_hits"],
                        "keyword_string_hits": crypto_report["keyword_string_hits"],
                        "cryptography_api_imports": crypto_report["cryptography_api_imports"],
                        "package_path_string_evidence": crypto_report["package_path_string_evidence"],
                        "compiled_function_symbol_string_evidence": {
                            key: value for key, value in crypto_report["compiled_function_symbol_string_evidence"].items()
                            if key != "symbols"
                        },
                        "finding_count": crypto_report["finding_count"],
                        "disassembly_call_sites_to_crypto_functions_or_imports": crypto_report["disassembly_call_sites_to_crypto_functions_or_imports"],
                        "go_call_graph": {key: value for key, value in crypto_report["go_direct_call_graph"].items()
                                          if key != "root_functions" and key != "limitations"},
                    },
                    "indirect_call_discovery": indirect_call_discovery,
                    "warnings": export_warnings,
                },
            }
            self._json_write(self.output_dir / "program.json", program_report)
            with (self.output_dir / "decompiler.jsonl").open("w", encoding="utf-8") as f:
                for row in decompiler_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary = {
                "input": self.input_path, "architecture": image.architecture, "bits": image.bits,
                "artifact_identity": crypto_report["artifact_identity"],
                "analysis_complete": not self.cancelled,
                "cancelled": self.cancelled,
                "cancelled_stage": self.cancelled_stage,
                "image_base": hex(image.image_base), "entry_point": hex(image.entry_point),
                "file_size_bytes": Path(self.input_path).stat().st_size,
                "section_count": len(image.sections),
                "sections": [{"name": s.name, "start": hex(s.start), "virtual_size": s.virtual_size,
                              "raw_size": s.raw_size, "executable": s.executable} for s in image.sections],
                "instruction_count": len(instructions), "basic_block_count": len(cfg.blocks),
                "function_candidates": len(functions), "pseudocode_functions_emitted": pseudocode_emitted_functions,
                "isolated_jump_thunk_candidates": [hex(x) for x in sorted(thunk_candidates)],
                "pseudocode_instructions_emitted": pseudocode_emitted_instructions,
                "function_signatures_recovered": len(signature_rows),
                "nonvoid_function_signatures": sum(
                    1 for row in signature_rows if row.get("return_type") not in {"void", None}
                ),
                "per_function_block_cap": self.max_blocks_per_function,
                "max_pseudocode_instructions": self.max_pseudocode_instructions, "string_count": len(strings),
                "crypto_indicator_summary": {
                    "analyzer": crypto_report["analyzer"],
                    "algorithm_names_found": crypto_report["algorithm_names_found"],
                    "algorithm_use_states": crypto_report["algorithm_use_states"],
                    "algorithm_families": crypto_report["algorithm_families"],
                    "go_function_symbol_names": crypto_report["algorithm_evidence_tiers"]["go_function_symbol_names"],
                    "algorithm_string_hits": crypto_report["algorithm_string_hits"],
                    "keyword_string_hits": crypto_report["keyword_string_hits"],
                    "cryptography_api_imports": crypto_report["cryptography_api_imports"],
                    "package_path_string_evidence": crypto_report["package_path_string_evidence"],
                    "compiled_function_symbol_string_evidence": {
                        key: value for key, value in crypto_report["compiled_function_symbol_string_evidence"].items()
                        if key != "symbols"
                    },
                    "finding_count": crypto_report["finding_count"],
                    "disassembly_call_sites_to_crypto_functions_or_imports": crypto_report["disassembly_call_sites_to_crypto_functions_or_imports"],
                    "go_call_graph": {key: value for key, value in crypto_report["go_direct_call_graph"].items()
                                      if key != "root_functions" and key != "limitations"},
                },
                "import_count": len(image.imports), "import_xref_count": sum(1 for r in xref_rows if r["kind"] == "import"),
                "string_xref_count": sum(1 for r in xref_rows if r["kind"] == "string"),
                "dataflow_functions_analyzed": dataflow_functions_count,
                "dataflow_rows": dataflow_rows_count,
                "dataflow_enabled": self.write_dataflow,
                "indirect_call_discovery": indirect_call_discovery,
                "static_pointer_scan": static_pointer_scan,
                "indirect_control_transfers_recorded": len(indirect_control_transfers),
                "normalized_ir_operations": normalized_ir_count,
                "ssa_operations": ssa_operation_count,
                "ssa_phi_nodes": ssa_phi_count,
                "ssa_functions_analyzed": ssa_function_count,
                "ssa_enabled": self.write_ssa,
                "max_dataflow_instructions": self.max_dataflow_instructions,
                "max_dataflow_functions": self.max_dataflow_functions,
                "windows_x64_runtime_function_records": len(image.runtime_functions),
                "windows_x64_exception_handler_targets": len(image.exception_handler_starts),
                "windows_x64_exception_handler_candidates": sum(
                    1 for function in functions
                    if "Windows exception/unwind handler metadata" in function.reason
                ),
                "runtime_function_records_with_handler": sum(
                    1 for record in image.runtime_functions if record.get("handler_va") is not None
                ),
                "unwind_metadata_records": sum(
                    1 for record in image.runtime_functions if record.get("unwind_version", -1) >= 0
                ),
                "executable_sections_scanned_for_strings": False,
                "code_discovery": {
                    "method": "recursive_control_flow_with_linear_sweep_candidates",
                    "recursive_seed_count": len(recursive_seeds),
                    "recursive_instruction_count": sum(
                        ins.discovery_source == "recursive_control_flow" for ins in instructions
                    ),
                    "linear_sweep_candidate_instruction_count": len(candidate_instructions),
                    "cfg_instruction_boundary_confidence": sorted({
                        ins.boundary_confidence for ins in instructions
                    }),
                    "application_root_function_count": 1 if image.entry_point in function_by_address else 0,
                    "application_reachable_function_count": len(application_reachable),
                    "unknown_application_reachability_function_count": sum(
                        state["reachable_from_application_root"] is None
                        for state in function_analysis_states.values()
                    ),
                },
                "output_files": ["program.json", "decompiler.jsonl"], "warnings": export_warnings,
            }
            if self.write_assembly:
                with (self.output_dir / "assembly.asm").open("w", encoding="utf-8") as f:
                    f.write("; Recursive control-flow decode is used by CFG/function analysis.\n")
                    f.write("; Linear-sweep candidates are listed separately and are not confirmed code.\n")
                    for ins in instructions:
                        f.write(ins.text + "\n")
                    f.write("\n; Linear-sweep candidate records (may include embedded data):\n")
                    confirmed_addresses = {item.address for item in instructions}
                    for ins in candidate_instructions:
                        if ins.address not in confirmed_addresses:
                            f.write("; candidate " + ins.text + "\n")
                summary["output_files"].append("assembly.asm")
            if self.write_ir:
                with (self.output_dir / "ir.jsonl").open("w", encoding="utf-8") as f:
                    for ins in instructions:
                        f.write(json.dumps({"address": hex(ins.address), "section": ins.section,
                                            "bytes": ins.bytes_hex, "mnemonic": ins.mnemonic,
                                            "operands": ins.operands, "flow": ins.flow,
                                            "direct_target": hex(ins.direct_target) if ins.direct_target is not None else None,
                                            "reference_target": hex(ins.reference_target) if ins.reference_target is not None else None,
                                            "indirect_targets": [hex(x) for x in (ins.indirect_targets or [])],
                                            "jump_table_entry_size": ins.jump_table_entry_size,
                                            "jump_table_base": hex(ins.jump_table_base) if ins.jump_table_base is not None else None,
                                            "jump_table_entries_observed": ins.jump_table_entries_observed,
                                            "jump_table_index_register": ins.jump_table_index_register,
                                            "jump_table_index_bias": ins.jump_table_index_bias,
                                            "jump_table_case_count": ins.jump_table_case_count,
                                            "jump_table_slots": [hex(target) if target is not None else None for target in (ins.jump_table_slots or [])],
                                            "jump_table_default_target": hex(ins.jump_table_default_target) if ins.jump_table_default_target is not None else None,
                                            "jump_table_default_entry_indices": ins.jump_table_default_entry_indices,
                                            "jump_table_bound_branch": ins.jump_table_bound_branch,
                                            "jump_table_bound_instruction": hex(ins.jump_table_bound_instruction) if ins.jump_table_bound_instruction is not None else None,
                                            "indirect_call_targets": [hex(x) for x in (ins.indirect_call_targets or [])],
                                            "resolved_call_provenance": ins.resolved_call_provenance,
                                            "unresolved_call_reason": ins.unresolved_call_reason,
                                            "discovery_source": ins.discovery_source,
                                            "boundary_confidence": ins.boundary_confidence,
                                            "import_name": ins.import_name}, ensure_ascii=False) + "\n")
                summary["output_files"].append("ir.jsonl")
            if self.write_ssa:
                summary["output_files"].extend(["normalized_ir.jsonl", "ssa_ir.jsonl", "ssa_cfg.json", "ssa_summary.json"])
            function_rows = []
            for function in functions:
                row = function.to_dict()
                row["analysis_states"] = function_analysis_states.get(
                    function.address, {"states": ["unknown"]}
                )
                function_rows.append(row)
            self._json_write(self.output_dir / "functions.json", function_rows)
            summary["output_files"].append("functions.json")
            summary["output_files"].append("function_signatures.json")
            self._json_write(self.output_dir / "cfg.json", [b.to_dict() for b in sorted(cfg.blocks.values(), key=lambda x: x.start)])
            summary["output_files"].append("cfg.json")
            self._json_write(self.output_dir / "strings.json", [s.to_dict() for s in strings])
            summary["output_files"].append("strings.json")
            self._json_write(self.output_dir / "crypto_findings.json", crypto_report)
            summary["output_files"].append("crypto_findings.json")
            self._json_write(
                self.output_dir / "crypto_cbom.cyclonedx.json",
                build_crypto_cbom(self.input_path, crypto_report),
            )
            summary["output_files"].append("crypto_cbom.cyclonedx.json")
            self._json_write(self.output_dir / "imports.json", [{"address": hex(a), "symbol": n} for a, n in sorted(image.imports.items())])
            summary["output_files"].append("imports.json")
            with (self.output_dir / "xrefs.csv").open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=["source", "target", "kind", "value"])
                w.writeheader(); w.writerows(xref_rows)
            summary["output_files"].append("xrefs.csv")
            summary["output_files"].append("pseudocode.c.txt")
            if self.write_dataflow:
                summary["output_files"].extend(["value_flow.jsonl", "dataflow_summary.json"])
            # Stage timings are part of the summary; they are instrumentation, not benchmark claims.
            summary["timings_seconds"] = {
                "load_pe_and_symbols": round(load_seconds, 6),
                "string_scan": round(string_scan_seconds, 6),
                "disassembly": round(decode_seconds, 6),
                "static_pointer_scan": round(pointer_scan_seconds, 6),
                "cfg_and_xrefs": round(cfg_seconds, 6),
                "dataflow": round(dataflow_seconds, 6),
                "normalized_ir_and_ssa": round(ir_seconds, 6),
                "output_export": round(time.perf_counter() - stage, 6),
                "total": round(time.perf_counter() - total_start, 6),
            }
            self._json_write(self.output_dir / "summary.json", summary)
            return summary
        finally:
            image.close()
