import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decompiler_engine.pipeline import DecompilerPipeline
from decompiler_engine.pe_loader import PEImage


def _minimal_pe64(code: bytes) -> bytes:
    """Build a tiny valid PE32+ image containing real x86-64 machine code."""
    pe_offset = 0x80
    file_alignment = 0x200
    section_alignment = 0x1000
    raw_size = (len(code) + file_alignment - 1) & ~(file_alignment - 1)
    headers_size = 0x200
    image_size = 0x2000

    dos = bytearray(pe_offset)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, pe_offset)
    coff = struct.pack(
        "<HHIIIHH", 0x8664, 1, 0, 0, 0, 0xF0, 0x0022,
    )
    optional = struct.pack(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        0x20B, 14, 0, raw_size, 0, 0, 0x1000, 0x1000,
        0x140000000, section_alignment, file_alignment,
        6, 0, 0, 0, 6, 0, 0, image_size, headers_size,
        0, 3, 0x8160, 0x100000, 0x1000, 0x100000, 0x1000,
        0, 16,
    ) + bytes(16 * 8)
    section = struct.pack(
        "<8sIIIIIIHHI", b".text\0\0\0", len(code), 0x1000,
        raw_size, headers_size, 0, 0, 0, 0, 0x60000020,
    )
    headers = bytes(dos) + b"PE\0\0" + coff + optional + section
    if len(headers) > headers_size:
        raise AssertionError("test PE headers exceed their declared size")
    headers += bytes(headers_size - len(headers))
    return headers + code + bytes(raw_size - len(code))


def _minimal_pe32(code: bytes) -> bytes:
    """Build a tiny valid PE32 image containing actual x86 machine code."""
    pe_offset = 0x80
    file_alignment = 0x200
    raw_size = (len(code) + file_alignment - 1) & ~(file_alignment - 1)
    headers_size = 0x200
    optional = struct.pack(
        "<HBBIIIIIIIIIHHHHHHIIIIHHIIIIII",
        0x10B, 14, 0, raw_size, 0, 0, 0x1000, 0x1000, 0,
        0x400000, 0x1000, file_alignment,
        6, 0, 0, 0, 6, 0, 0, 0x2000, headers_size,
        0, 3, 0x8140, 0x100000, 0x1000, 0x100000, 0x1000,
        0, 16,
    ) + bytes(16 * 8)
    dos = bytearray(pe_offset)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, pe_offset)
    coff = struct.pack("<HHIIIHH", 0x14C, 1, 0, 0, 0, len(optional), 0x0102)
    section = struct.pack(
        "<8sIIIIIIHHI", b".text\0\0\0", len(code), 0x1000,
        raw_size, headers_size, 0, 0, 0, 0, 0x60000020,
    )
    headers = bytes(dos) + b"PE\0\0" + coff + optional + section
    if len(headers) > headers_size:
        raise AssertionError("test PE headers exceed their declared size")
    headers += bytes(headers_size - len(headers))
    return headers + code + bytes(raw_size - len(code))


def _pe64_with_import_and_unwind() -> bytes:
    """Build a PE32+ image with a real IAT import and x64 unwind record."""
    image_base = 0x140000000
    headers_size = 0x400
    section_raw_size = 0x200
    iat_rva = 0x2050
    entry = image_base + 0x1000
    call_end = entry + 6
    code = b"\xff\x15" + struct.pack("<i", image_base + iat_rva - call_end) + b"\xc3"

    directories = bytearray(16 * 8)
    struct.pack_into("<II", directories, 8, 0x2000, 40)  # import directory
    struct.pack_into("<II", directories, 24, 0x3000, 12)  # exception directory
    optional = struct.pack(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        0x20B, 14, 0, section_raw_size, section_raw_size * 3, 0,
        0x1000, 0x1000, image_base, 0x1000, 0x200,
        6, 0, 0, 0, 6, 0, 0, 0x5000, headers_size,
        0, 3, 0x8160, 0x100000, 0x1000, 0x100000, 0x1000,
        0, 16,
    ) + bytes(directories)
    dos = bytearray(0x80)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x80)
    coff = struct.pack("<HHIIIHH", 0x8664, 4, 0, 0, 0, len(optional), 0x0022)
    section_specs = [
        (b".text", len(code), 0x1000, headers_size, 0x60000020),
        (b".rdata", 0x200, 0x2000, headers_size + 0x200, 0x40000040),
        (b".pdata", 12, 0x3000, headers_size + 0x400, 0x40000040),
        (b".xdata", 4, 0x4000, headers_size + 0x600, 0x40000040),
    ]
    section_headers = b"".join(
        struct.pack(
            "<8sIIIIIIHHI", name.ljust(8, b"\0"), virtual_size,
            rva, section_raw_size, raw_offset, 0, 0, 0, 0, characteristics,
        )
        for name, virtual_size, rva, raw_offset, characteristics in section_specs
    )
    headers = dos + b"PE\0\0" + coff + optional + section_headers
    if len(headers) > headers_size:
        raise AssertionError("test PE headers exceed their declared size")
    headers += bytes(headers_size - len(headers))

    rdata = bytearray(section_raw_size)
    # IMAGE_IMPORT_DESCRIPTOR, then a null descriptor.
    struct.pack_into("<IIIII", rdata, 0, 0x2040, 0, 0, 0x2080, iat_rva)
    struct.pack_into("<II", rdata, 0x40, 0x2090, 0)
    struct.pack_into("<II", rdata, 0x50, 0x2090, 0)
    rdata[0x80:0x80 + len(b"KERNEL32.dll\0")] = b"KERNEL32.dll\0"
    rdata[0x90:0x90 + len(b"\0\0GetLastError\0")] = b"\0\0GetLastError\0"
    pdata = struct.pack("<III", 0x1000, 0x1000 + len(code), 0x4000)
    xdata = bytes([1, 0, 0, 0])  # version 1 UNWIND_INFO with no prolog codes
    return (
        bytes(headers) + code + bytes(section_raw_size - len(code))
        + bytes(rdata) + pdata + bytes(section_raw_size - len(pdata))
        + xdata + bytes(section_raw_size - len(xdata))
    )


def test_minimal_real_pe64_runs_through_loader_decoder_and_pipeline(tmp_path):
    # mov eax, 42; ret
    image_path = tmp_path / "minimal_x64.exe"
    image_path.write_bytes(_minimal_pe64(bytes.fromhex("B8 2A 00 00 00 C3")))

    image = PEImage(image_path)
    try:
        assert image.architecture == "x86-64"
        assert image.entry_point == image.image_base + 0x1000
        assert image.section_for(image.entry_point).executable
    finally:
        image.close()

    output = tmp_path / "analysis"
    summary = DecompilerPipeline(
        str(image_path), str(output), write_assembly=False, write_ir=False,
    ).run()

    assert summary["architecture"] == "x86-64"
    assert summary["instruction_count"] == 2
    assert summary["function_candidates"] == 1
    assert (output / "functions.json").is_file()
    pseudocode = (output / "pseudocode.c.txt").read_text(encoding="utf-8")
    assert "return 42;" in pseudocode.lower()


def test_minimal_real_pe32_runs_through_loader_decoder_and_pipeline(tmp_path):
    image_path = tmp_path / "minimal_x86.exe"
    image_path.write_bytes(_minimal_pe32(bytes.fromhex("B8 2A 00 00 00 C3")))

    image = PEImage(image_path)
    try:
        assert image.architecture == "x86"
        assert image.entry_point == image.image_base + 0x1000
        assert image.section_for(image.entry_point).executable
    finally:
        image.close()

    output = tmp_path / "analysis_x86"
    summary = DecompilerPipeline(
        str(image_path), str(output), write_assembly=False, write_ir=False,
    ).run()
    assert summary["architecture"] == "x86"
    assert summary["instruction_count"] == 2
    assert summary["function_candidates"] == 1
    assert (output / "functions.json").is_file()


def test_real_pe64_import_and_unwind_metadata_reach_pipeline_reports(tmp_path):
    image_path = tmp_path / "import_unwind_x64.exe"
    image_path.write_bytes(_pe64_with_import_and_unwind())
    image = PEImage(image_path)
    try:
        iat = image.image_base + 0x2050
        assert image.imports[iat] == "KERNEL32.dll!GetLastError"
        assert image.runtime_function_starts == {image.image_base + 0x1000}
        assert image.runtime_functions[0]["start"] == image.image_base + 0x1000
        assert image.runtime_functions[0]["end"] == image.image_base + 0x1007
        assert image.runtime_functions[0]["unwind_version"] == 1
    finally:
        image.close()

    output = tmp_path / "analysis_import_unwind"
    summary = DecompilerPipeline(
        str(image_path), str(output), write_assembly=False, write_ir=False,
    ).run()
    assert summary["architecture"] == "x86-64"
    assert summary["import_count"] == 1
    assert summary["windows_x64_runtime_function_records"] == 1
    report = json.loads((output / "program.json").read_text(encoding="utf-8"))
    assert report["symbols"]["imports"] == [{
        "address": "0x140002050", "name": "KERNEL32.dll!GetLastError", "external": True,
    }]
    assert report["runtime_function_unwind"][0]["start"] == "0x140001000"
    assert report["runtime_function_unwind"][0]["end"] == "0x140001007"
    assert report["runtime_function_unwind"][0]["version"] == 1
    pseudocode = (output / "pseudocode.c.txt").read_text(encoding="utf-8")
    assert "GetLastError()" in pseudocode
