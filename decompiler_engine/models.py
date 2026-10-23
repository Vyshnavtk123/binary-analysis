from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

@dataclass(slots=True)
class Instruction:
    address: int
    size: int
    bytes_hex: str
    mnemonic: str
    operands: str
    section: str = ""
    direct_target: int | None = None
    reference_target: int | None = None
    flow: str = "normal"  # normal, call, conditional, jump, return
    import_name: str | None = None
    indirect_targets: list[int] | None = None
    jump_table_entry_size: int | None = None
    jump_table_base: int | None = None
    indirect_call_targets: list[int] | None = None
    resolved_call_provenance: str | None = None
    unresolved_call_reason: str | None = None
    jump_table_entries_observed: int | None = None
    jump_table_index_register: str | None = None
    jump_table_case_count: int | None = None
    jump_table_bound_branch: str | None = None
    jump_table_bound_instruction: int | None = None
    jump_table_index_bias: int | None = None
    jump_table_default_target: int | None = None
    jump_table_default_entry_indices: list[int] | None = None
    jump_table_slots: list[int | None] | None = None
    resolved_indirect_jump_target: int | None = None
    discovery_source: str = "linear_sweep"
    boundary_confidence: str = "candidate"

    @property
    def end(self) -> int:
        return self.address + self.size

    @property
    def text(self) -> str:
        return f"0x{self.address:016X}: {self.mnemonic:<8} {self.operands}".rstrip()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["address"] = hex(self.address)
        d["direct_target"] = hex(self.direct_target) if self.direct_target is not None else None
        d["reference_target"] = hex(self.reference_target) if self.reference_target is not None else None
        d["indirect_targets"] = [hex(x) for x in (self.indirect_targets or [])]
        d["jump_table_slots"] = [hex(x) if x is not None else None for x in (self.jump_table_slots or [])]
        d["resolved_indirect_jump_target"] = (
            hex(self.resolved_indirect_jump_target)
            if self.resolved_indirect_jump_target is not None else None
        )
        d["jump_table_base"] = hex(self.jump_table_base) if self.jump_table_base is not None else None
        d["jump_table_entries_observed"] = self.jump_table_entries_observed
        d["jump_table_bound_instruction"] = (
            hex(self.jump_table_bound_instruction)
            if self.jump_table_bound_instruction is not None else None
        )
        d["jump_table_default_target"] = (
            hex(self.jump_table_default_target)
            if self.jump_table_default_target is not None else None
        )
        d["indirect_call_targets"] = [hex(x) for x in (self.indirect_call_targets or [])]
        d["resolved_call_provenance"] = self.resolved_call_provenance
        d["unresolved_call_reason"] = self.unresolved_call_reason
        d["end"] = hex(self.end)
        return d

@dataclass(slots=True)
class BasicBlock:
    start: int
    end: int
    instructions: list[Instruction]
    successors: list[int]
    section: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": hex(self.start), "end": hex(self.end), "section": self.section,
            "instruction_count": len(self.instructions),
            "first_instruction": self.instructions[0].text if self.instructions else None,
            "last_instruction": self.instructions[-1].text if self.instructions else None,
            "successors": [hex(x) for x in self.successors],
        }

@dataclass(slots=True)
class FunctionInfo:
    address: int
    name: str
    reason: str
    block_starts: list[int]
    calls: list[int]
    strings: list[dict[str, Any]]
    instruction_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": hex(self.address), "name": self.name, "reason": self.reason,
            "blocks": [hex(x) for x in self.block_starts],
            "calls": [hex(x) for x in self.calls], "strings": self.strings,
            "instruction_count": self.instruction_count,
        }
