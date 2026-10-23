from __future__ import annotations

"""Bounded SSA-style renaming for the independent normalized IR.

Implements iterative dominators, immediate dominators, dominance frontiers, phi
placement, and dominator-tree renaming for registers and simple frame-relative slots.
Memory aliasing and full x86 semantics are intentionally outside this first pass.
"""

from collections import defaultdict
from copy import deepcopy
from typing import Any

_TRACK_PREFIXES = ("RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "RBP", "RSP", "RIP", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15", "STACK_")
_KNOWN_TYPES = {"uint8_t", "uint16_t", "uint32_t", "uint64_t", "uintptr_t",
                "bool", "flags_t", "vector128_t", "vector256_t", "vector512_t",
                "stack_value_t", "int", "void"}


def _is_var(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith(("const:", "address:", "MEM[", "opaque:")):
        return False
    return value == "FLAGS" or value.startswith(_TRACK_PREFIXES)


def _merge_type(left: str | None, right: str | None) -> str:
    if left == right and left in _KNOWN_TYPES:
        return left
    return "unknown"


def _dominators(nodes: list[str], entry: str, preds: dict[str, set[str]]) -> dict[str, set[str]]:
    all_nodes = set(nodes)
    dom = {n: ({n} if n == entry else set(all_nodes)) for n in nodes}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n == entry:
                continue
            ps = preds.get(n, set())
            common = set.intersection(*(dom[p] for p in ps)) if ps else set()
            new = common | {n}
            if new != dom[n]:
                dom[n] = new
                changed = True
    return dom


def _idom_tree(nodes: list[str], entry: str, dom: dict[str, set[str]]) -> tuple[dict[str, str | None], dict[str, list[str]]]:
    idom: dict[str, str | None] = {entry: None}
    for n in nodes:
        if n == entry:
            continue
        strict = dom[n] - {n}
        candidate = None
        for d in strict:
            # The immediate dominator is the strict dominator not dominating any
            # other strict dominator candidate closer to this node.
            if not any(d != other and d in dom[other] for other in strict):
                candidate = d
                break
        if candidate is None and strict:
            candidate = max(strict, key=lambda x: len(dom[x]))
        idom[n] = candidate
    children: dict[str, list[str]] = {n: [] for n in nodes}
    for child, parent in idom.items():
        if parent is not None:
            children[parent].append(child)
    for c in children.values(): c.sort()
    return idom, children


def _dominance_frontier(nodes: list[str], preds: dict[str, set[str]], idom: dict[str, str | None]) -> dict[str, set[str]]:
    frontier = {n: set() for n in nodes}
    for b in nodes:
        if len(preds.get(b, set())) < 2:
            continue
        for pred in preds[b]:
            runner: str | None = pred
            hops = 0
            while runner is not None and runner != idom.get(b) and hops <= len(nodes):
                frontier[runner].add(b)
                runner = idom.get(runner)
                hops += 1
    return frontier


def construct_ssa(blocks: list[Any], ir_rows: list[dict[str, Any]], entry_address: int | str) -> dict[str, Any]:
    """Return SSA-transformed IR and CFG metadata for one candidate function."""
    if not blocks:
        return {"entry": str(entry_address), "blocks": [], "operations": [], "summary": {"phi_count": 0, "tracked_variables": 0}}
    block_by_id = {hex(b.start): b for b in blocks}
    nodes = sorted(block_by_id, key=lambda s: int(s, 16))
    entry = hex(entry_address) if isinstance(entry_address, int) else str(entry_address)
    if entry not in block_by_id:
        entry = nodes[0]
    allowed = set(nodes)
    preds: dict[str, set[str]] = {n: set() for n in nodes}
    for block_id, block in block_by_id.items():
        for s in block.successors:
            ss = hex(s)
            if ss in allowed:
                preds[ss].add(block_id)
    # Unreachable components shouldn't normally occur (input blocks are reachable),
    # but keep dominator computation robust when an input is malformed.
    dom = _dominators(nodes, entry, preds)
    idom, children = _idom_tree(nodes, entry, dom)
    df = _dominance_frontier(nodes, preds, idom)

    original: dict[str, list[dict[str, Any]]] = {n: [] for n in nodes}
    def_blocks: dict[str, set[str]] = defaultdict(set)
    all_vars: set[str] = set()
    row_by_block = {n: [] for n in nodes}
    for row in ir_rows:
        block_id = row.get("block")
        if block_id not in allowed:
            continue
        copied = deepcopy(row)
        original[block_id].append(copied)
        row_by_block[block_id].append(copied)
        for use in copied.get("inputs", []):
            if _is_var(use): all_vars.add(use)
        for definition in copied.get("defs", []):
            if _is_var(definition):
                all_vars.add(definition)
                def_blocks[definition].add(block_id)

    # Iterated dominance-frontier algorithm for placing phi nodes.
    phis: dict[str, dict[str, dict[str, Any]]] = {n: {} for n in nodes}
    for var in sorted(def_blocks):
        work = list(def_blocks[var])
        visited: set[str] = set()
        while work:
            block_id = work.pop()
            for frontier_block in sorted(df[block_id]):
                if frontier_block in visited:
                    continue
                visited.add(frontier_block)
                phis[frontier_block][var] = {
                    "address": None, "address_int": None, "block": frontier_block,
                    "mnemonic": "phi", "assembly_operands": "", "opcode": "phi",
                    "destination": var, "defs": [var], "inputs": [], "incoming": {},
                    "expression": "phi", "width": None, "flow": "normal", "target": None,
                    "reference_target": None, "import_name": None, "modeled": True,
                    "input_types": [], "destination_type": None,
                }
                if frontier_block not in def_blocks[var]:
                    work.append(frontier_block)

    version_count: dict[str, int] = defaultdict(int)
    stacks: dict[str, list[str]] = {v: [f"{v}_0"] for v in all_vars}
    base_types: dict[str, str] = {}
    for row in (row for rows in original.values() for row in rows):
        input_types = row.get("input_types", [])
        for value, type_name in zip(row.get("inputs", []), input_types):
            if _is_var(value) and isinstance(type_name, str):
                base_types[value] = _merge_type(base_types.get(value), type_name) if value in base_types else type_name
        destination = row.get("destination")
        destination_type = row.get("destination_type")
        if isinstance(destination, str) and _is_var(destination) and isinstance(destination_type, str):
            base_types[destination] = _merge_type(base_types.get(destination), destination_type) if destination in base_types else destination_type
    version_types: dict[str, str] = {}
    emitted: dict[str, list[dict[str, Any]]] = {n: [] for n in nodes}
    phi_count = 0

    def current(var: str) -> str:
        return stacks.setdefault(var, [f"{var}_0"])[-1]

    def new_version(var: str) -> str:
        version_count[var] += 1
        name = f"{var}_{version_count[var]}"
        stacks.setdefault(var, [f"{var}_0"]).append(name)
        return name

    def rename(block_id: str) -> None:
        nonlocal phi_count
        pushed: list[str] = []
        # Phi results define the variable at the start of this block.
        for var, phi in sorted(phis[block_id].items()):
            dst = new_version(var)
            phi["destination_ssa"] = dst
            phi["defs_ssa"] = [dst]
            phi_type = base_types.get(var, "unknown")
            phi["destination_type"] = phi_type
            phi["destination_type_ssa"] = phi_type
            version_types[dst] = phi_type
            emitted[block_id].append(phi)
            pushed.append(var)
            phi_count += 1
        for row in original[block_id]:
            new_row = row
            renamed_inputs: list[str] = []
            renamed_input_types: list[str] = []
            substitutions: dict[str, str] = {}
            for index, value in enumerate(new_row.get("inputs", [])):
                renamed_value = current(value) if _is_var(value) else value
                renamed_inputs.append(renamed_value)
                if _is_var(value):
                    renamed_input_types.append(version_types.get(renamed_value, base_types.get(value, "unknown")))
                else:
                    input_types = new_row.get("input_types", [])
                    renamed_input_types.append(input_types[index] if index < len(input_types) else "unknown")
                if _is_var(value):
                    substitutions[value] = renamed_value
            new_row["inputs_ssa"] = renamed_inputs
            new_row["input_types_ssa"] = renamed_input_types
            expression = new_row.get("expression")
            if isinstance(expression, str) and expression:
                for source_name in sorted(substitutions, key=len, reverse=True):
                    import re
                    expression = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(source_name)}(?![A-Za-z0-9_])",
                                        substitutions[source_name], expression)
                new_row["expression_ssa"] = expression
            else:
                new_row["expression_ssa"] = expression
            renamed_defs: list[str] = []
            renamed_def_types: list[str] = []
            for var in new_row.get("defs", []):
                if _is_var(var):
                    renamed = new_version(var)
                    renamed_defs.append(renamed)
                    definition_type = new_row.get("destination_type") if not renamed_def_types else None
                    definition_type = definition_type if isinstance(definition_type, str) else base_types.get(var, "unknown")
                    renamed_def_types.append(definition_type)
                    version_types[renamed] = definition_type
                    pushed.append(var)
                else:
                    renamed_defs.append(var)
                    renamed_def_types.append("unknown")
            new_row["defs_ssa"] = renamed_defs
            new_row["defs_types_ssa"] = renamed_def_types
            new_row["destination_ssa"] = renamed_defs[0] if renamed_defs else None
            new_row["destination_type_ssa"] = renamed_def_types[0] if renamed_def_types else None
            new_row["uses_ssa"] = [current(value) if _is_var(value) else value for value in new_row.get("inputs", [])]
            emitted[block_id].append(new_row)
        # Record values that reach each successor's phi from this predecessor.
        for succ in sorted(block_by_id[block_id].successors):
            succ_id = hex(succ)
            if succ_id not in allowed:
                continue
            for var, phi in phis[succ_id].items():
                phi.setdefault("incoming", {})[block_id] = current(var)
        for child in children.get(block_id, []):
            rename(child)
        for var in reversed(pushed):
            if stacks.get(var):
                stacks[var].pop()
                if not stacks[var]:
                    stacks[var] = [f"{var}_0"]

    rename(entry)
    # Keep phi inputs stable and explicit, including predecessors whose incoming edge
    # may be unresolved because of inconsistent CFG input.
    flat_ops: list[dict[str, Any]] = []
    out_blocks: list[dict[str, Any]] = []
    for block_id in nodes:
        ops = emitted[block_id]
        for op in ops:
            if op.get("opcode") == "phi":
                op["incoming"] = dict(sorted(op.get("incoming", {}).items()))
                op["inputs_ssa"] = list(op["incoming"].values())
                op["input_types_ssa"] = [
                    version_types.get(value, "unknown") for value in op["inputs_ssa"]
                ]
                if op["input_types_ssa"]:
                    phi_type = op["input_types_ssa"][0]
                    for input_type in op["input_types_ssa"][1:]:
                        phi_type = _merge_type(phi_type, input_type)
                    op["destination_type"] = phi_type
                    op["destination_type_ssa"] = phi_type
            # Input tokens aren't needed in SSA export and can be verbose.
            flat_ops.append({k: v for k, v in op.items() if k != "input_tokens"})
        out_blocks.append({
            "start": block_id,
            "predecessors": sorted(preds[block_id]),
            "successors": [hex(x) for x in block_by_id[block_id].successors if hex(x) in allowed],
            "immediate_dominator": idom[block_id],
            "dominance_frontier": sorted(df[block_id]),
            "phi_count": len(phis[block_id]),
        })
    summary = {
        "block_count": len(nodes), "operation_count": len(flat_ops),
        "phi_count": phi_count, "tracked_variables": len(all_vars),
        "variables": sorted(all_vars),
        "typed_versions": dict(sorted(version_types.items())),
        "typed_version_count": sum(type_name != "unknown" for type_name in version_types.values()),
        "dominators_computed": True, "ssa_renaming_completed": True,
        "memory_aliasing_modeled": False,
    }
    return {"entry": entry, "blocks": out_blocks, "operations": flat_ops, "summary": summary}
