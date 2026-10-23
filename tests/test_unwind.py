import struct
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decompiler_engine.pe_loader import PEImage


def test_unwind_info_decodes_alloc_push_frame_and_handler():
    data = bytes([
        9, 7, 4, 0x25,   # version 1 + EHANDLER; prolog; four slots; RBP frame, scale 2
        7, 0x32,         # UWOP_ALLOC_SMALL: 32 bytes
        4, 0x50,         # UWOP_PUSH_NONVOL: RBP
        2, 0x01,         # UWOP_ALLOC_LARGE, OpInfo 0
        2, 0,            # extra slot: 2 * 8 bytes
    ]) + struct.pack("<I", 0x5000)

    parsed = PEImage.parse_unwind_info(data)
    assert parsed["unwind_version"] == 1
    assert parsed["unwind_flag_names"] == ["EHANDLER"]
    assert parsed["frame_register_name"] == "rbp"
    assert parsed["frame_register_scale"] == 2
    assert [op["operation"] for op in parsed["unwind_operations"]] == [
        "UWOP_ALLOC_SMALL", "UWOP_PUSH_NONVOL", "UWOP_ALLOC_LARGE",
    ]
    assert parsed["unwind_operations"][0]["allocation_size"] == 32
    assert parsed["unwind_operations"][1]["register"] == "rbp"
    assert parsed["unwind_operations"][2]["allocation_size"] == 16
    assert parsed["handler_rva"] == 0x5000
    assert parsed["language_data_offset"] == 16
    assert parsed["unwind_parse_warnings"] == []


def test_unwind_info_decodes_chained_runtime_function_record():
    data = bytes([33, 0, 0, 0]) + struct.pack("<III", 0x1000, 0x1100, 0x2200)
    parsed = PEImage.parse_unwind_info(data)
    assert parsed["unwind_flag_names"] == ["CHAININFO"]
    assert parsed["chained_begin_rva"] == 0x1000
    assert parsed["chained_end_rva"] == 0x1100
    assert parsed["chained_unwind_info_rva"] == 0x2200
    assert parsed["unwind_parse_warnings"] == []


def test_unwind_info_decodes_saved_registers_frame_and_machine_frame_ops():
    slots = bytearray()
    encoded = [
        (9, 4, 2, struct.pack("<H", 3)),       # SAVE_NONVOL RDX, offset 24
        (8, 5, 1, struct.pack("<I", 0x40)),    # SAVE_NONVOL_FAR RCX, offset 64
        (7, 8, 6, struct.pack("<H", 2)),       # SAVE_XMM128 XMM6, offset 32
        (6, 9, 7, struct.pack("<I", 0x30)),    # SAVE_XMM128_FAR XMM7
        (5, 3, 0, b""),                        # SET_FPREG
        (4, 10, 1, b""),                       # PUSH_MACHFRAME with error code
    ]
    slot_count = 0
    for code_offset, operation, op_info, extra in encoded:
        slots.extend((code_offset, (op_info << 4) | operation))
        slots.extend(extra)
        slot_count += 1 + len(extra) // 2
    data = bytes([1, 9, slot_count, 0x25]) + bytes(slots)

    parsed = PEImage.parse_unwind_info(data)
    operations = parsed["unwind_operations"]
    assert operations[0]["register"] == "rdx"
    assert operations[0]["stack_offset"] == 24
    assert operations[1]["register"] == "rcx"
    assert operations[1]["stack_offset"] == 64
    assert operations[2]["register"] == "xmm6"
    assert operations[2]["stack_offset"] == 32
    assert operations[3]["register"] == "xmm7"
    assert operations[3]["stack_offset"] == 0x30
    assert operations[4]["frame_register"] == "rbp"
    assert operations[4]["frame_offset"] == 32
    assert operations[5]["machine_frame_error_code"] is True
    assert operations[5]["machine_frame_size"] == 48
    assert parsed["unwind_parse_warnings"] == []


def test_version_two_epilog_opcode_is_accepted():
    parsed = PEImage.parse_unwind_info(bytes([2, 3, 1, 0, 2, 0x06, 0, 0]))
    assert parsed["unwind_operations"][0]["operation"] == "UWOP_EPILOG"
    assert parsed["unwind_parse_warnings"] == []


def test_unwind_info_marks_truncated_operands_without_overreading():
    data = bytes([1, 2, 1, 0, 2, 0x01, 0, 0])
    parsed = PEImage.parse_unwind_info(data)
    assert any("exceeds the declared UNWIND_CODE count" in warning
               for warning in parsed["unwind_parse_warnings"])
    assert parsed["handler_rva"] is None


def test_unwind_info_reports_truncated_header():
    parsed = PEImage.parse_unwind_info(b"\x01\x02")
    assert parsed["unwind_version"] == -1
    assert parsed["unwind_parse_warnings"] == ["UNWIND_INFO header is truncated."]


def test_runtime_function_loader_discovers_executable_handler_entry():
    image_base = 0x140000000
    code = SimpleNamespace(name=".text", start=image_base + 0x1000, end=image_base + 0x2000, executable=True)
    pdata = SimpleNamespace(name=".pdata", start=image_base + 0x5000, end=image_base + 0x5010)
    xdata = SimpleNamespace(name=".xdata", start=image_base + 0x6000, end=image_base + 0x6010, executable=False)
    handler_rva = 0x1800
    unwind = bytes([9, 0, 0, 0]) + struct.pack("<I", handler_rva)
    fake = SimpleNamespace(
        bits=64,
        image_base=image_base,
        parse_unwind_info=PEImage.parse_unwind_info,
        sections=[code, pdata, xdata],
        runtime_functions=[],
        runtime_function_starts=set(),
        exception_handler_starts=set(),
        pe=SimpleNamespace(),
        section_bytes=lambda section: struct.pack("<III", 0x1000, 0x1010, 0x6000)
        if section.name == ".pdata" else b"",
        read_va=lambda address, limit=4096: unwind if address == image_base + 0x6000 else b"",
        section_for=lambda address: next(
            (section for section in [code, pdata, xdata] if section.start <= address < section.end), None
        ),
    )

    PEImage._parse_runtime_functions(fake)
    assert fake.runtime_function_starts == {image_base + 0x1000}
    assert fake.exception_handler_starts == {image_base + handler_rva}
    assert fake.runtime_functions[0]["handler_va"] == image_base + handler_rva
