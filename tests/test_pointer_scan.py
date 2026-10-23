import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decompiler_engine.pe_loader import PEImage, Section


def test_bounded_pointer_scan_keeps_aligned_readable_data_pointers_into_image():
    image = object.__new__(PEImage)
    image.bits = 64
    image.sections = [
        Section(".text", 0x1000, 0x1100, 0x100, 32, 0, True, True, False),
        Section(".rdata", 0x5000, 0x5018, 0x18, 0, 0x18, False, True, False),
    ]
    image._data = struct.pack("<QQQ", 0x1010, 0xDEAD, 0x1080)
    pointers, stats = image.scan_pointer_slots(max_slots=10)
    assert pointers == {0x5000: 0x1010, 0x5010: 0x1080}
    assert stats == {
        "slots_scanned": 3,
        "pointer_slots_found": 2,
        "slot_budget": 10,
        "budget_exhausted": False,
    }


def test_pointer_scan_reports_budget_exhaustion_without_scanning_extra_slots():
    image = object.__new__(PEImage)
    image.bits = 64
    image.sections = [Section(".data", 0x6000, 0x6020, 0x20, 0, 0x20, False, True, True)]
    image._data = struct.pack("<QQQQ", 0x6008, 0x6000, 0x6008, 0x6000)
    pointers, stats = image.scan_pointer_slots(max_slots=2)
    assert pointers == {0x6000: 0x6008, 0x6008: 0x6000}
    assert stats["slots_scanned"] == 2
    assert stats["budget_exhausted"] is True
