from types import SimpleNamespace

from decompiler_engine.disassembly import Disassembler


def test_recursive_decode_excludes_linear_sweep_bytes_skipped_by_jump():
    # jmp +2; two NOP bytes that linear sweep sees; ret at the branch target.
    section = SimpleNamespace(start=0x1000, end=0x1005, executable=True, name=".text")
    image = SimpleNamespace(
        bits=64, sections=[section], imports={}, exports={},
        section_bytes=lambda _section: bytes.fromhex("EB 02 90 90 C3"),
    )
    decoder = Disassembler(image)

    candidates = decoder.run()
    recursive = decoder.run_recursive({0x1000}, candidates)

    assert [ins.address for ins in candidates] == [0x1000, 0x1002, 0x1003, 0x1004]
    assert [ins.address for ins in recursive] == [0x1000, 0x1004]
    assert all(ins.boundary_confidence == "confirmed_reachable" for ins in recursive)
    assert all(ins.discovery_source == "recursive_control_flow" for ins in recursive)


def test_recursive_decode_follows_conditional_and_direct_call_targets():
    # jz +5; call +1; ret; ret. Calls and conditional branches enqueue targets.
    section = SimpleNamespace(start=0x2000, end=0x2009, executable=True, name=".text")
    image = SimpleNamespace(
        bits=64, sections=[section], imports={}, exports={},
        section_bytes=lambda _section: bytes.fromhex("74 05 E8 01 00 00 00 C3 C3"),
    )
    decoder = Disassembler(image)
    recursive = decoder.run_recursive({0x2000})

    starts = {ins.address for ins in recursive}
    assert 0x2007 in starts  # conditional target
    assert 0x2008 in starts  # direct call target
    assert all(ins.boundary_confidence == "confirmed_reachable" for ins in recursive)


def test_recursive_decode_accepts_valid_target_that_was_not_a_seed():
    # call +3 reaches the final RET; the middle RET is fallthrough after the call.
    section = SimpleNamespace(start=0x3000, end=0x3009, executable=True, name=".text")
    image = SimpleNamespace(
        bits=64, sections=[section], imports={}, exports={},
        section_bytes=lambda _section: bytes.fromhex("E8 03 00 00 00 C3 90 90 C3"),
    )
    decoder = Disassembler(image)
    recursive = decoder.run_recursive({0x3000})
    assert {ins.address for ins in recursive} >= {0x3000, 0x3005, 0x3008}


def test_recursive_decode_rejects_target_inside_existing_instruction():
    section = SimpleNamespace(start=0x4000, end=0x4005, executable=True, name=".text")
    image = SimpleNamespace(
        bits=64, sections=[section], imports={}, exports={},
        section_bytes=lambda _section: bytes.fromhex("48 89 C0 C3 90"),
    )
    decoder = Disassembler(image)
    existing = decoder.run_recursive({0x4000})
    expanded = decoder.run_recursive({0x4001}, existing_instructions=existing)
    assert 0x4001 not in {ins.address for ins in expanded}


def test_recursive_decode_rejects_unbacked_executable_virtual_bytes():
    section = SimpleNamespace(start=0x5000, end=0x5010, executable=True, name=".text")
    image = SimpleNamespace(
        bits=64, sections=[section], imports={}, exports={},
        section_bytes=lambda _section: bytes.fromhex("C3"),
    )
    decoder = Disassembler(image)
    recursive = decoder.run_recursive({0x5001})
    assert recursive == []


def test_recursive_decode_prioritizes_application_root_when_instruction_cap_is_tight():
    section = SimpleNamespace(start=0x6000, end=0x7001, executable=True, name=".text")
    image = SimpleNamespace(
        bits=64, sections=[section], imports={}, exports={},
        section_bytes=lambda _section: bytes.fromhex("C3") + bytes(0xFFF) + bytes.fromhex("C3"),
    )
    decoder = Disassembler(image, max_instructions=1)
    decoded = decoder.run_recursive({0x6000, 0x7000}, priority_seeds={0x7000})
    assert [ins.address for ins in decoded] == [0x7000]
