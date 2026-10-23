import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decompiler_engine.models import Instruction
from decompiler_engine.cfg import ControlFlowGraph
from decompiler_engine.disassembly import Disassembler

def inst(address, size, mnemonic, target=None, flow="normal", indirect_targets=None):
    return Instruction(address, size, "90", mnemonic, "", ".text", target, None, flow, None, indirect_targets)

def test_conditional_branch_has_taken_and_fallthrough_edges():
    ins = [
        inst(0x1000, 2, "cmp"),
        inst(0x1002, 2, "jne", target=0x1008, flow="conditional"),
        inst(0x1004, 1, "ret", flow="return"),
        inst(0x1008, 1, "ret", flow="return"),
    ]
    cfg = ControlFlowGraph(ins)
    assert cfg.blocks[0x1000].successors == [0x1008, 0x1004]
    assert cfg.blocks[0x1004].successors == []
    assert cfg.blocks[0x1008].successors == []

def test_reachability_stops_at_other_function_entry():
    ins = [
        inst(0x1000, 1, "call", target=0x1010, flow="call"),
        inst(0x1001, 1, "ret", flow="return"),
        inst(0x1010, 1, "ret", flow="return"),
    ]
    cfg = ControlFlowGraph(ins)
    blocks = cfg.reachable_blocks(0x1000, {0x1000, 0x1010})
    assert blocks == [0x1000]
    assert [x.address for x in cfg.blocks[0x1000].instructions] == [0x1000, 0x1001]
    assert 0x1010 not in blocks


def test_explicit_function_entry_splits_linear_block():
    ins = [
        inst(0x2000, 1, "nop"),
        inst(0x2001, 1, "nop"),
        inst(0x2002, 1, "ret", flow="return"),
    ]
    cfg = ControlFlowGraph(ins, extra_leaders={0x2001})
    assert 0x2001 in cfg.blocks
    assert cfg.reachable_blocks(0x2001, {0x2001}) == [0x2001]
    assert [x.address for x in cfg.blocks[0x2001].instructions] == [0x2001, 0x2002]


def test_indirect_jump_targets_create_cfg_edges():
    ins = [
        inst(0x3000, 2, "jmp", flow="jump", indirect_targets=[0x3010, 0x3020]),
        inst(0x3010, 1, "ret", flow="return"),
        inst(0x3020, 1, "ret", flow="return"),
    ]
    cfg = ControlFlowGraph(ins)
    assert cfg.blocks[0x3000].successors == [0x3010, 0x3020]


def test_x64_relative_32bit_jump_table_entries_are_recovered():
    class FakeSection:
        def __init__(self, start, end):
            self.start = start
            self.end = end
            self.executable = True

    class FakeImage:
        bits = 64
        image_base = 0x10000000
        sections = [FakeSection(0x1000, 0x2000)]

        def read_va(self, address, size):
            values = [0x100, 0x200]
            return b"".join(value.to_bytes(4, "little", signed=True) for value in values)[:size]

        def section_for(self, address):
            return self.sections[0] if self.sections[0].start <= address < self.sections[0].end else None

    disassembler = object.__new__(Disassembler)
    disassembler.image = FakeImage()
    targets, entry_size = disassembler._read_jump_table(0x1000)
    assert targets == [0x1100, 0x1200]
    assert entry_size == 4


def test_jump_table_scan_stops_at_first_invalid_entry():
    class FakeSection:
        start = 0x1000
        end = 0x2000
        executable = True

    class FakeImage:
        bits = 64
        image_base = 0x10000000
        sections = [FakeSection()]

        def read_va(self, address, size):
            values = [0xFFFFFFFF, 0x100, 0x200]
            return b"".join(value.to_bytes(4, "little") for value in values)[:size]

        def section_for(self, address):
            return self.sections[0] if 0x1000 <= address < 0x2000 else None

    disassembler = object.__new__(Disassembler)
    disassembler.image = FakeImage()
    targets, entry_size, observed = disassembler._read_jump_table_details(0x1000)
    assert targets == []
    assert entry_size is None
    assert observed == 0


def test_bounded_sparse_jump_table_retains_invalid_case_slots():
    class FakeSection:
        start = 0x1000
        end = 0x2000
        executable = True

    class FakeImage:
        bits = 64
        image_base = 0x10000000
        sections = [FakeSection()]

        def read_va(self, address, size):
            values = [0x100, 0xFFFFFFFF, 0x200, 0xFFFFFFFF]
            return b"".join(value.to_bytes(4, "little") for value in values)[:size]

        def section_for(self, address):
            return self.sections[0] if 0x1000 <= address < 0x2000 else None

    disassembler = object.__new__(Disassembler)
    disassembler.image = FakeImage()
    targets, entry_size, observed, slots = disassembler._read_jump_table_slots(
        0x1000, 4, allow_sparse=True
    )
    assert targets == [0x1100, 0x1200]
    assert entry_size == 4
    assert observed == 2
    assert slots == [0x1100, None, 0x1200, None]


def test_disassembler_uses_guard_bound_to_report_sparse_table_layout():
    from capstone import CS_OP_IMM, CS_OP_MEM
    from capstone.x86_const import X86_REG_RIP

    class FakeSection:
        name = ".text"
        start = 0x1000
        end = 0x3000
        executable = True

    class FakeOperand:
        def __init__(self, kind, *, imm=0, base=0, disp=0):
            self.type = kind
            self.imm = imm
            self.mem = type("Mem", (), {"base": base, "disp": disp})()

    class FakeDecoded:
        def __init__(self, address, size, mnemonic, op_str, operands):
            self.address = address
            self.size = size
            self.mnemonic = mnemonic
            self.op_str = op_str
            self.operands = operands
            self.bytes = b"\x90" * size

    decoded = [
        FakeDecoded(0x1000, 2, "cmp", "ecx, 3", []),
        FakeDecoded(0x1002, 2, "ja", "0x1010", [FakeOperand(CS_OP_IMM, imm=0x1010)]),
        FakeDecoded(
            0x1004, 6, "jmp", "dword ptr [rip + rcx*4 + 0xff6]",
            [FakeOperand(CS_OP_MEM, base=X86_REG_RIP, disp=0xFF6)],
        ),
    ]

    class FakeDecoder:
        def disasm(self, _blob, _address):
            return decoded

    class FakeImage:
        bits = 64
        image_base = 0x10000000
        sections = [FakeSection()]
        imports = {}
        exports = {}

        def section_bytes(self, _section):
            return b"\x90" * 0x2000

        def read_va(self, address, size):
            if address != 0x2000:
                return b""
            values = [0x100, 0xFFFFFFFF, 0x200, 0xFFFFFFFF]
            return b"".join(value.to_bytes(4, "little") for value in values)[:size]

        def section_for(self, address):
            return self.sections[0] if 0x1000 <= address < 0x3000 and address != 0x1FFF else None

    disassembler = object.__new__(Disassembler)
    disassembler.image = FakeImage()
    disassembler.md = FakeDecoder()
    disassembler.max_instructions = 100
    disassembler.warnings = []
    jump = next(ins for ins in disassembler.run() if ins.mnemonic == "jmp")
    assert jump.jump_table_case_count == 4
    assert jump.indirect_targets == [0x2100, 0x2200]
    assert jump.jump_table_slots == [0x2100, None, 0x2200, None]


def test_disassembler_uses_signed_range_guards_to_bound_sparse_jump_table():
    from capstone import CS_OP_IMM, CS_OP_MEM
    from capstone.x86_const import X86_REG_RIP

    class FakeSection:
        name = ".text"
        start = 0x1000
        end = 0x3000
        executable = True

    class FakeOperand:
        def __init__(self, kind, *, imm=0, base=0, disp=0):
            self.type = kind
            self.imm = imm
            self.mem = type("Mem", (), {"base": base, "disp": disp})()

    class FakeDecoded:
        def __init__(self, address, size, mnemonic, op_str, operands):
            self.address, self.size = address, size
            self.mnemonic, self.op_str = mnemonic, op_str
            self.operands, self.bytes = operands, b"\x90" * size

    decoded = [
        FakeDecoded(0x1000, 2, "test", "ecx, ecx", []),
        FakeDecoded(0x1002, 2, "js", "0x1018", [FakeOperand(CS_OP_IMM, imm=0x1018)]),
        FakeDecoded(0x1004, 2, "cmp", "ecx, 2", []),
        FakeDecoded(0x1006, 2, "jg", "0x1018", [FakeOperand(CS_OP_IMM, imm=0x1018)]),
        FakeDecoded(
            0x1008, 6, "jmp", "dword ptr [rip + rcx*4 + 0xff2]",
            [FakeOperand(CS_OP_MEM, base=X86_REG_RIP, disp=0xFF2)],
        ),
    ]

    class FakeDecoder:
        def disasm(self, _blob, _address):
            return decoded

    class FakeImage:
        bits = 64
        image_base = 0x10000000
        sections = [FakeSection()]
        imports = {}
        exports = {}

        def section_bytes(self, _section):
            return b"\x90" * 0x2000

        def read_va(self, address, size):
            if address != 0x2000:
                return b""
            values = [0x100, 0xFFFFFFFF, 0x200]
            return b"".join(value.to_bytes(4, "little") for value in values)[:size]

        def section_for(self, address):
            return self.sections[0] if 0x1000 <= address < 0x3000 and address != 0x1FFF else None

    disassembler = object.__new__(Disassembler)
    disassembler.image = FakeImage()
    disassembler.md = FakeDecoder()
    disassembler.max_instructions = 100
    disassembler.warnings = []
    jump = next(ins for ins in disassembler.run() if ins.mnemonic == "jmp")
    assert jump.jump_table_case_count == 3
    assert jump.jump_table_bound_branch == "jg"
    assert jump.jump_table_slots == [0x2100, None, 0x2200]


def test_disassembler_resolves_indexed_table_base_from_rip_relative_lea():
    from capstone import CS_OP_IMM, CS_OP_MEM
    from capstone.x86_const import X86_REG_RIP

    class FakeSection:
        name = ".text"
        start = 0x1000
        end = 0x3000
        executable = True

    class FakeOperand:
        def __init__(self, kind, *, imm=0, base=0, disp=0):
            self.type = kind
            self.imm = imm
            self.mem = type("Mem", (), {"base": base, "disp": disp})()

    class FakeDecoded:
        def __init__(self, address, size, mnemonic, op_str, operands):
            self.address, self.size = address, size
            self.mnemonic, self.op_str = mnemonic, op_str
            self.operands, self.bytes = operands, b"\x90" * size

    decoded = [
        FakeDecoded(
            0x1000, 7, "lea", "rax, [rip + 0xff9]",
            [FakeOperand(CS_OP_MEM, base=X86_REG_RIP, disp=0xFF9)],
        ),
        FakeDecoded(0x1007, 2, "cmp", "ecx, 2", []),
        FakeDecoded(0x1009, 2, "ja", "0x1018", [FakeOperand(CS_OP_IMM, imm=0x1018)]),
        FakeDecoded(
            0x100B, 6, "jmp", "dword ptr [rax + rcx*4]",
            [FakeOperand(CS_OP_MEM, base=0, disp=0)],
        ),
    ]

    class FakeDecoder:
        def disasm(self, _blob, _address):
            return decoded

    class FakeImage:
        bits = 64
        image_base = 0x10000000
        sections = [FakeSection()]
        imports = {}
        exports = {}

        def section_bytes(self, _section):
            return b"\x90" * 0x2000

        def read_va(self, address, size):
            if address != 0x2000:
                return b""
            values = [0x100, 0xFFFFFFFF, 0x200]
            return b"".join(value.to_bytes(4, "little") for value in values)[:size]

        def section_for(self, address):
            return self.sections[0] if 0x1000 <= address < 0x3000 and address != 0x1FFF else None

    disassembler = object.__new__(Disassembler)
    disassembler.image = FakeImage()
    disassembler.md = FakeDecoder()
    disassembler.max_instructions = 100
    disassembler.warnings = []
    jump = next(ins for ins in disassembler.run() if ins.mnemonic == "jmp")
    assert jump.reference_target == 0x2000
    assert jump.jump_table_case_count == 3
    assert jump.jump_table_slots == [0x2100, None, 0x2200]


def test_disassembler_recovers_register_based_signed_relative_jump_table():
    from capstone import CS_OP_IMM, CS_OP_MEM
    from capstone.x86_const import X86_REG_RIP

    class FakeSection:
        name, start, end, executable = ".text", 0x1000, 0x3000, True

    class FakeOperand:
        def __init__(self, kind, *, imm=0, base=0, disp=0):
            self.type, self.imm = kind, imm
            self.mem = type("Mem", (), {"base": base, "disp": disp})()

    class FakeDecoded:
        def __init__(self, address, size, mnemonic, op_str, operands=()):
            self.address, self.size, self.mnemonic, self.op_str = address, size, mnemonic, op_str
            self.operands, self.bytes = list(operands), b"\x90" * size

    decoded = [
        FakeDecoded(0x1000, 7, "lea", "rdx, [rip + 0xff9]",
                    [FakeOperand(CS_OP_MEM, base=X86_REG_RIP, disp=0xFF9)]),
        FakeDecoded(0x1007, 2, "cmp", "ecx, 2"),
        FakeDecoded(0x1009, 2, "ja", "0x1018", [FakeOperand(CS_OP_IMM, imm=0x1018)]),
        FakeDecoded(0x100B, 6, "movsxd", "rax, dword ptr [rdx + rcx*4]"),
        FakeDecoded(0x1011, 3, "add", "rax, rdx"),
        FakeDecoded(0x1014, 2, "jmp", "rax"),
    ]

    class FakeDecoder:
        def disasm(self, _blob, _address):
            return decoded

    class FakeImage:
        bits, image_base = 64, 0x10000000
        sections, imports, exports = [FakeSection()], {}, {}

        def section_bytes(self, _section):
            return b"\x90" * 0x2000

        def read_va(self, address, size):
            if address != 0x2000:
                return b""
            return b"".join(value.to_bytes(4, "little") for value in (0x100, 0xFFFFFFFF, 0x200))[:size]

        def section_for(self, address):
            return self.sections[0] if 0x1000 <= address < 0x3000 and address != 0x1FFF else None

    disassembler = object.__new__(Disassembler)
    disassembler.image, disassembler.md = FakeImage(), FakeDecoder()
    disassembler.max_instructions, disassembler.warnings = 100, []
    jump = next(ins for ins in disassembler.run() if ins.mnemonic == "jmp")
    assert jump.reference_target == 0x2000
    assert jump.jump_table_index_register == "rcx"
    assert jump.jump_table_case_count == 3
    assert jump.jump_table_slots == [0x2100, None, 0x2200]


def test_disassembler_resolves_bounded_indexed_indirect_call_table():
    from capstone import CS_OP_IMM, CS_OP_MEM
    from capstone.x86_const import X86_REG_RIP

    class FakeSection:
        name, start, end, executable = ".text", 0x1000, 0x3000, True

    class FakeOperand:
        def __init__(self, kind, *, imm=0, base=0, disp=0):
            self.type, self.imm = kind, imm
            self.mem = type("Mem", (), {"base": base, "disp": disp})()

    class FakeDecoded:
        def __init__(self, address, size, mnemonic, op_str, operands=()):
            self.address, self.size = address, size
            self.mnemonic, self.op_str = mnemonic, op_str
            self.operands, self.bytes = list(operands), b"\x90" * size

    decoded = [
        FakeDecoded(0x1000, 2, "cmp", "ecx, 1"),
        FakeDecoded(0x1002, 2, "ja", "0x1018", [FakeOperand(CS_OP_IMM, imm=0x1018)]),
        FakeDecoded(0x1004, 6, "call", "qword ptr [rip + rcx*8 + 0xff6]",
                    [FakeOperand(CS_OP_MEM, base=X86_REG_RIP, disp=0xFF6)]),
        FakeDecoded(0x100A, 1, "ret", ""),
        FakeDecoded(0x2100, 1, "ret", ""),
        FakeDecoded(0x2200, 1, "ret", ""),
    ]

    class FakeDecoder:
        def disasm(self, _blob, _address):
            return decoded

    class FakeImage:
        bits, image_base = 64, 0x10000000
        sections, imports, exports = [FakeSection()], {}, {}

        def section_bytes(self, _section):
            return b"\x90" * 0x2000

        def read_va(self, address, size):
            if address != 0x2000:
                return b""
            return b"".join(value.to_bytes(8, "little") for value in (0x2100, 0x2200))[:size]

        def section_for(self, address):
            return self.sections[0] if 0x1000 <= address < 0x3000 else None

    disassembler = object.__new__(Disassembler)
    disassembler.image, disassembler.md = FakeImage(), FakeDecoder()
    disassembler.max_instructions, disassembler.warnings = 100, []
    call = next(ins for ins in disassembler.run() if ins.mnemonic == "call")
    assert call.reference_target == 0x2000
    assert call.jump_table_case_count == 2
    assert call.indirect_call_targets == [0x2100, 0x2200]


def test_jump_table_reports_contiguous_entries_observed_including_default_duplicates():
    class FakeSection:
        start = 0x1000
        end = 0x2000
        executable = True

    class FakeImage:
        bits = 64
        image_base = 0x10000000
        sections = [FakeSection()]

        def read_va(self, address, size):
            values = [0x100, 0x100, 0x200, 0xFFFFFFFF]
            return b"".join(value.to_bytes(4, "little") for value in values)[:size]

        def section_for(self, address):
            return self.sections[0] if 0x1000 <= address < 0x2000 else None

    disassembler = object.__new__(Disassembler)
    disassembler.image = FakeImage()
    targets, entry_size, observed = disassembler._read_jump_table_details(0x1000)
    assert targets == [0x1100, 0x1200]
    assert entry_size == 4
    assert observed == 3
