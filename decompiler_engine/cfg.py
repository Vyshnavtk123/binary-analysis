from __future__ import annotations
from collections import deque
from .models import Instruction, BasicBlock

class ControlFlowGraph:
    def __init__(self, instructions: list[Instruction], extra_leaders: set[int] | None = None):
        self.instructions = instructions
        self.by_address = {x.address: x for x in instructions}
        self.extra_leaders = set(extra_leaders or ())
        self.blocks: dict[int, BasicBlock] = {}
        self.block_for_instruction: dict[int, int] = {}
        self._build()

    def _build(self) -> None:
        if not self.instructions: return
        addrs = [i.address for i in self.instructions]
        addr_set = set(addrs)
        # Entry points and metadata-derived function starts must split a block even
        # when they are not also direct branch targets (e.g., a PE entry thunk).
        leaders = {addrs[0]} | (self.extra_leaders & addr_set)
        for i in self.instructions:
            # Each direct control-transfer target starts its own block when aligned.
            if i.direct_target in addr_set:
                leaders.add(i.direct_target)
            for target in i.indirect_targets or ():
                if target in addr_set:
                    leaders.add(target)
        for i, ins in enumerate(self.instructions):
            nxt = self.instructions[i + 1] if i + 1 < len(self.instructions) else None
            if nxt and (nxt.address != ins.end or nxt.section != ins.section):
                leaders.add(nxt.address)
            if ins.flow in ("conditional", "jump", "return") and nxt:
                leaders.add(nxt.address)
        current: list[Instruction] = []
        for ins in self.instructions:
            if current and (ins.address in leaders or current[-1].end != ins.address or current[-1].section != ins.section):
                self._finish_block(current)
                current = []
            current.append(ins)
        if current: self._finish_block(current)
        starts = set(self.blocks)
        for start, block in self.blocks.items():
            last = block.instructions[-1]
            succ: list[int] = []
            next_addr = last.end
            next_ins = self.by_address.get(next_addr)
            if last.flow == "conditional":
                if last.direct_target in starts: succ.append(last.direct_target)
                if next_ins and next_ins.section == last.section and next_addr in starts and next_addr not in succ:
                    succ.append(next_addr)
            elif last.flow == "jump":
                if last.direct_target in starts: succ.append(last.direct_target)
                for target in last.indirect_targets or ():
                    if target in starts and target not in succ:
                        succ.append(target)
            elif last.flow == "return":
                pass
            else:
                if next_ins and next_ins.section == last.section and next_addr in starts:
                    succ.append(next_addr)
            block.successors = succ

    def _finish_block(self, insns: list[Instruction]) -> None:
        start, end = insns[0].address, insns[-1].end
        block = BasicBlock(start, end, insns[:], [], insns[0].section)
        self.blocks[start] = block
        for ins in insns: self.block_for_instruction[ins.address] = start

    def reachable_blocks(self, entry: int, function_entries: set[int] | None = None, max_blocks: int = 2_000) -> list[int]:
        """Return reachable intraprocedural blocks; stop at other known function entries."""
        if entry not in self.blocks: return []
        fn_entries = function_entries or set()
        queue = deque([entry])
        seen: set[int] = set()
        while queue and len(seen) < max_blocks:
            cur = queue.popleft()
            if cur in seen: continue
            if cur != entry and cur in fn_entries: continue
            if cur not in self.blocks: continue
            seen.add(cur)
            for nxt in self.blocks[cur].successors:
                if nxt not in seen and (nxt == entry or nxt not in fn_entries): queue.append(nxt)
        return sorted(seen)
