from __future__ import annotations
import mmap
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pefile

@dataclass(slots=True)
class Section:
    name: str
    start: int
    end: int
    virtual_size: int
    raw_offset: int
    raw_size: int
    executable: bool
    readable: bool
    writable: bool

    @property
    def backed_size(self) -> int:
        span = self.virtual_size if self.virtual_size > 0 else self.raw_size
        return min(span, self.raw_size)

@dataclass(slots=True)
class BinaryString:
    address: int
    text: str
    encoding: str
    section: str
    length: int

    def to_dict(self) -> dict[str, Any]:
        return {"address": hex(self.address), "text": self.text, "encoding": self.encoding,
                "section": self.section, "length": self.length}

class PEImage:
    """Read-only, memory-mapped Windows PE loader for x86/x64 images."""
    def __init__(self, path: str | Path):
        self.path = str(Path(path).resolve())
        self.pe = pefile.PE(self.path, fast_load=True)
        self._file = open(self.path, "rb")
        try:
            self._data = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            machine = int(self.pe.FILE_HEADER.Machine)
            if machine == 0x8664:
                self.architecture, self.bits = "x86-64", 64
            elif machine == 0x14C:
                self.architecture, self.bits = "x86", 32
            else:
                raise ValueError(f"Unsupported PE machine 0x{machine:X}; supported: x86 and x86-64")
            self.image_base = int(self.pe.OPTIONAL_HEADER.ImageBase)
            self.entry_point = self.image_base + int(self.pe.OPTIONAL_HEADER.AddressOfEntryPoint)
            self.sections: list[Section] = []
            for sec in self.pe.sections:
                vsize = int(sec.Misc_VirtualSize)
                raw_size = int(sec.SizeOfRawData)
                flags = int(sec.Characteristics)
                name = sec.Name.rstrip(b"\0").decode("ascii", errors="replace")
                start = self.image_base + int(sec.VirtualAddress)
                self.sections.append(Section(
                    name=name, start=start, end=start + max(vsize, raw_size),
                    virtual_size=vsize, raw_offset=int(sec.PointerToRawData), raw_size=raw_size,
                    executable=bool(flags & 0x20000000), readable=bool(flags & 0x40000000),
                    writable=bool(flags & 0x80000000)))
            self.imports: dict[int, str] = {}
            self.exports: dict[int, str] = {}
            self.runtime_functions: list[dict[str, Any]] = []
            self.runtime_function_starts: set[int] = set()
            self.exception_handler_starts: set[int] = set()
            self.go_functions: dict[int, str] = {}
            self.go_symbol_metadata: dict[str, Any] = {"format": None, "function_count": 0, "warnings": []}
            self.go_build_metadata: dict[str, Any] = {"available": False, "toolchain": None, "module": None,
                                                      "dependencies": [], "build_settings": [], "warnings": []}
            self._parse_symbols()
            self._parse_runtime_functions()
            self._parse_go_build_info()
            self._parse_go_pclntab()
        except Exception:
            try: self._data.close()
            except Exception: pass
            self._file.close()
            self.pe.close()
            raise

    def _parse_go_build_info(self) -> None:
        """Read Go 1.18+ inline build information (toolchain and module records)."""
        marker = b"\xff Go buildinf:"
        base = self._data.find(marker)
        if base < 0 or base + 16 > len(self._data):
            self.go_build_metadata["warnings"].append("Go build-info header was not found.")
            return
        pointer_size, flags = self._data[base + 14], self._data[base + 15]
        if pointer_size not in (4, 8) or not (flags & 0x02):
            self.go_build_metadata["warnings"].append("Go build-info encoding is unsupported (expected inline strings).")
            return
        cursor = base + 16 + 2 * pointer_size

        def read_uvarint(offset: int) -> tuple[int, int]:
            value = shift = 0
            while offset < len(self._data) and shift < 64:
                byte = self._data[offset]
                offset += 1
                value |= (byte & 0x7f) << shift
                if byte < 0x80:
                    return value, offset
                shift += 7
            raise ValueError("malformed Go build-info varint")

        try:
            version_size, cursor = read_uvarint(cursor)
            if version_size > 256 or cursor + version_size > len(self._data):
                raise ValueError("Go toolchain version length is invalid")
            version = bytes(self._data[cursor:cursor + version_size]).decode("utf-8", errors="replace")
            cursor += version_size
            module_size, cursor = read_uvarint(cursor)
            if module_size > 16 * 1024 * 1024 or cursor + module_size > len(self._data):
                raise ValueError("Go module metadata length is invalid")
            module_text = bytes(self._data[cursor:cursor + module_size]).decode("utf-8", errors="replace")
            module = None
            dependencies = []
            settings = []
            for line in module_text.splitlines():
                fields = line.split("\t")
                if fields[0] == "path" and len(fields) > 1:
                    self.go_build_metadata["main_package"] = fields[1]
                elif fields[0] == "mod" and len(fields) > 1:
                    module = {"path": fields[1], "version": fields[2] if len(fields) > 2 else None,
                              "sum": fields[3] if len(fields) > 3 else None}
                elif fields[0] == "dep" and len(fields) > 1:
                    dependencies.append({"path": fields[1], "version": fields[2] if len(fields) > 2 else None,
                                         "sum": fields[3] if len(fields) > 3 else None})
                elif fields[0] == "=>" and len(fields) > 1:
                    replacement = {"path": fields[1], "version": fields[2] if len(fields) > 2 else None,
                                   "sum": fields[3] if len(fields) > 3 else None}
                    if dependencies:
                        dependencies[-1]["replace"] = replacement
                    elif module is not None:
                        module["replace"] = replacement
                elif fields[0] == "build" and len(fields) > 1:
                    settings.append({"key": fields[1], "value": fields[2] if len(fields) > 2 else ""})
            self.go_build_metadata.update({"available": True, "toolchain": version, "module": module,
                                           "dependencies": dependencies, "build_settings": settings,
                                           "format": "Go inline build info"})
        except (ValueError, OverflowError) as exc:
            self.go_build_metadata["warnings"].append(f"Go build-info parse failed safely: {exc}")

    def _parse_symbols(self) -> None:
        directories = []
        for name in ("IMAGE_DIRECTORY_ENTRY_IMPORT", "IMAGE_DIRECTORY_ENTRY_EXPORT"):
            idx = pefile.DIRECTORY_ENTRY.get(name)
            if idx is not None:
                directories.append(idx)
        try:
            self.pe.parse_data_directories(directories=directories)
        except Exception:
            pass
        for desc in getattr(self.pe, "DIRECTORY_ENTRY_IMPORT", []):
            dll = desc.dll.decode("ascii", errors="replace") if desc.dll else "unknown.dll"
            for imp in desc.imports:
                name = imp.name.decode("ascii", errors="replace") if imp.name else f"ordinal_{imp.ordinal}"
                self.imports[int(imp.address)] = f"{dll}!{name}"
        export_dir = getattr(self.pe, "DIRECTORY_ENTRY_EXPORT", None)
        if export_dir:
            for sym in export_dir.symbols:
                try:
                    addr = int(sym.address)
                    va = addr if addr >= self.image_base else self.image_base + addr
                    name = sym.name.decode("ascii", errors="replace") if sym.name else f"ordinal_{sym.ordinal}"
                    self.exports[va] = name
                except Exception:
                    continue

    def _parse_go_pclntab(self) -> None:
        """Recover Go 1.18+ x64 function entry PCs/names from the PE pclntab."""
        if self.bits != 64:
            self.go_symbol_metadata["warnings"].append(
                "Go pclntab recovery currently supports 64-bit PE images only; inline metadata is not parsed."
            )
            return
        data = self._data
        magics = {b"\xf0\xff\xff\xff": "go1.18", b"\xf1\xff\xff\xff": "go1.20+"}
        base = -1
        magic_name = None
        cursor = 0
        while cursor < len(data) - 32:
            pos = data.find(b"\xf1\xff\xff\xff", cursor)
            pos18 = data.find(b"\xf0\xff\xff\xff", cursor)
            candidates = [x for x in (pos, pos18) if x >= 0]
            if not candidates:
                break
            candidate = min(candidates)
            cursor = candidate + 1
            magic = bytes(data[candidate:candidate + 4])
            if bytes(data[candidate + 4:candidate + 8]) != b"\0\0\1\x08":
                continue
            base, magic_name = candidate, magics[magic]
            break
        if base < 0:
            self.go_symbol_metadata["warnings"].append("Supported Go pclntab signature was not found.")
            return
        try:
            words = struct.unpack_from("<8Q", data, base + 8)
            function_count, _, text_start, funcname_offset, _, _, _, funcdata_offset = words
            if not (0 < function_count <= 1_000_000):
                raise ValueError("Go pclntab function count is outside the supported bound.")
            if text_start == 0:
                # Recent Go PE linkers can leave the pclntab textStart field zero
                # in the on-disk image. Function entry offsets remain relative to
                # the executable text section and must be rebased before mapping.
                text_section = next((section for section in self.sections if section.executable), None)
                if text_section is None:
                    raise ValueError("Go pclntab has no text start and the PE has no executable section.")
                text_start = text_section.start
                self.go_symbol_metadata["text_start_recovered_from_pe"] = True
            table_start = base + funcdata_offset
            table_size = (function_count * 2 + 1) * 4
            funcname_start = base + funcname_offset
            if table_start < base or table_start + table_size > len(data) or funcname_start >= len(data):
                raise ValueError("Go pclntab offsets are outside the file.")
            functions: dict[int, str] = {}
            previous_pc = 0
            for index in range(function_count):
                entry_offset, function_offset = struct.unpack_from("<II", data, table_start + index * 8)
                entry_pc = text_start + entry_offset
                if entry_pc <= previous_pc:
                    raise ValueError("Go pclntab function PCs are not strictly increasing.")
                previous_pc = entry_pc
                record = table_start + function_offset
                if record < table_start or record + 8 > len(data):
                    continue
                name_offset = struct.unpack_from("<i", data, record + 4)[0]
                name_pos = funcname_start + name_offset
                if name_offset < 0 or name_pos < funcname_start or name_pos >= len(data):
                    continue
                name_end = data.find(b"\0", name_pos, min(len(data), name_pos + 2048))
                if name_end < 0:
                    continue
                name = bytes(data[name_pos:name_end]).decode("utf-8", errors="replace")
                if not name or any(ord(ch) < 0x20 for ch in name):
                    continue
                section = self.section_for(entry_pc)
                if section is not None and section.executable:
                    functions[entry_pc] = name
            self.go_functions = functions
            self.go_symbol_metadata.update({
                "format": magic_name, "function_count": len(functions),
                "pclntab_file_offset": hex(base),
                "function_names_recovered": True,
                "inline_metadata_available": False,
                "limitations": ["Go pclntab function names are recovered for supported Go 1.18+ x64 layouts.",
                                "Inline-function trees and source line tables are not decoded.",
                                "Missing pclntab or unsupported compiler layouts leave symbols unresolved."],
            })
        except (struct.error, ValueError, OverflowError) as exc:
            self.go_functions = {}
            self.go_symbol_metadata["warnings"].append(f"Go pclntab parse failed safely: {exc}")

    def _parse_runtime_functions(self) -> None:
        """Read Windows x64 RUNTIME_FUNCTION records from the PE exception directory.

        Each record is three 32-bit RVAs (begin, end, unwind-info). These starts are
        useful evidence for function boundaries, but metadata may be absent or unusual.
        """
        if self.bits != 64:
            return
        raw = b""
        try:
            directory_index = pefile.DIRECTORY_ENTRY.get("IMAGE_DIRECTORY_ENTRY_EXCEPTION")
            if directory_index is not None:
                directory = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[directory_index]
                rva = int(directory.VirtualAddress)
                size = int(directory.Size)
                if rva and size >= 12:
                    raw = self.read_va(self.image_base + rva, size)
        except Exception:
            raw = b""
        if not raw:
            pdata = next((sec for sec in self.sections if sec.name == ".pdata"), None)
            if pdata is not None:
                raw = self.section_bytes(pdata)
        if len(raw) < 12:
            return
        for offset in range(0, len(raw) - 11, 12):
            begin_rva, end_rva, unwind_rva = struct.unpack_from("<III", raw, offset)
            if begin_rva == 0 and end_rva == 0 and unwind_rva == 0:
                continue
            if begin_rva >= end_rva:
                continue
            start = self.image_base + begin_rva
            end = self.image_base + end_rva
            start_section = self.section_for(start)
            end_section = self.section_for(end - 1)
            if not start_section or not start_section.executable:
                continue
            if end_section is not start_section:
                continue
            unwind_info = self.image_base + unwind_rva
            unwind = self.parse_unwind_info(self.read_va(unwind_info, 1024))
            record: dict[str, Any] = {
                "start": start, "end": end, "unwind_info": unwind_info, **unwind,
            }
            handler_rva = unwind.get("handler_rva")
            if isinstance(handler_rva, int) and handler_rva > 0:
                handler_va = self.image_base + handler_rva
                record["handler_va"] = handler_va
                handler_section = self.section_for(handler_va)
                if handler_section and handler_section.executable:
                    self.exception_handler_starts.add(handler_va)
            chain_begin = unwind.get("chained_begin_rva")
            if isinstance(chain_begin, int) and chain_begin > 0:
                chained_start = self.image_base + chain_begin
                chained_end = self.image_base + int(unwind["chained_end_rva"])
                record["chained_start"] = chained_start
                record["chained_end"] = chained_end
                record["chained_unwind_info"] = self.image_base + int(unwind["chained_unwind_info_rva"])
                chained_section = self.section_for(chained_start)
                if (chained_start < chained_end and chained_section and chained_section.executable
                        and self.section_for(chained_end - 1) is chained_section):
                    self.runtime_function_starts.add(chained_start)
            language_data_offset = unwind.get("language_data_offset")
            if isinstance(language_data_offset, int):
                record["language_data_va"] = unwind_info + language_data_offset
            self.runtime_functions.append(record)
            self.runtime_function_starts.add(start)

    @staticmethod
    def parse_unwind_info(data: bytes) -> dict[str, Any]:
        """Decode bounded Windows x64 UNWIND_INFO codes and trailing references.

        The result is metadata for analysis, not an emulator for Windows unwinding.
        Unknown/reserved opcodes and truncated data are preserved as warnings.
        """
        empty: dict[str, Any] = {
            "unwind_version": -1, "unwind_flags": 0, "prolog_size": -1,
            "unwind_code_count": -1, "frame_register": 0,
            "frame_register_scale": 0, "frame_register_name": None,
            "unwind_operations": [], "handler_rva": None,
            "language_data_offset": None, "chained_begin_rva": None,
            "chained_end_rva": None, "chained_unwind_info_rva": None,
            "unwind_parse_warnings": ["UNWIND_INFO header is truncated."],
        }
        if len(data) < 4:
            return empty

        version_flags, prolog_size, code_count, frame = data[:4]
        version = version_flags & 0x7
        flags = version_flags >> 3
        frame_register = frame & 0x0F
        frame_scale = frame >> 4
        register_names = {
            0: "rax", 1: "rcx", 2: "rdx", 3: "rbx", 4: "rsp", 5: "rbp",
            6: "rsi", 7: "rdi", 8: "r8", 9: "r9", 10: "r10", 11: "r11",
            12: "r12", 13: "r13", 14: "r14", 15: "r15",
        }
        result: dict[str, Any] = {
            "unwind_version": version,
            "unwind_flags": flags,
            "unwind_flag_names": [
                name for mask, name in ((1, "EHANDLER"), (2, "UHANDLER"), (4, "CHAININFO"))
                if flags & mask
            ],
            "prolog_size": prolog_size,
            "unwind_code_count": code_count,
            "frame_register": frame_register,
            "frame_register_scale": frame_scale,
            "frame_register_name": None if frame_register == 0 else register_names[frame_register],
            "unwind_operations": [],
            "handler_rva": None,
            "language_data_offset": None,
            "chained_begin_rva": None,
            "chained_end_rva": None,
            "chained_unwind_info_rva": None,
            "unwind_parse_warnings": [],
        }
        if version not in {1, 2}:
            result["unwind_parse_warnings"].append(f"Unsupported UNWIND_INFO version {version}.")
        if (flags & 0x7) not in {0, 1, 2, 4}:
            result["unwind_parse_warnings"].append("Conflicting UNWIND_INFO flags are set.")

        op_names = {
            0: "UWOP_PUSH_NONVOL", 1: "UWOP_ALLOC_LARGE", 2: "UWOP_ALLOC_SMALL",
            3: "UWOP_SET_FPREG", 4: "UWOP_SAVE_NONVOL", 5: "UWOP_SAVE_NONVOL_FAR",
            6: "UWOP_EPILOG", 7: "UWOP_SPARE_CODE", 8: "UWOP_SAVE_XMM128",
            9: "UWOP_SAVE_XMM128_FAR", 10: "UWOP_PUSH_MACHFRAME",
        }
        slot = 0
        while slot < code_count:
            offset = 4 + slot * 2
            if offset + 2 > len(data):
                result["unwind_parse_warnings"].append("UNWIND_CODE array is truncated.")
                break
            code_offset, op_byte = data[offset], data[offset + 1]
            operation = op_byte & 0x0F
            op_info = op_byte >> 4
            record: dict[str, Any] = {
                "code_offset": code_offset,
                "operation_code": operation,
                "operation": op_names.get(operation, f"UWOP_UNKNOWN_{operation}"),
                "op_info": op_info,
                "slot_index": slot,
            }
            extra_slots = 0
            if operation == 1:  # UWOP_ALLOC_LARGE
                extra_slots = 1 if op_info == 0 else 2 if op_info == 1 else 0
                if op_info not in {0, 1}:
                    result["unwind_parse_warnings"].append("UWOP_ALLOC_LARGE has an invalid OpInfo.")
            elif operation in {4, 8}:  # SAVE_NONVOL / SAVE_XMM128
                extra_slots = 1
            elif operation in {5, 9}:  # SAVE_NONVOL_FAR / SAVE_XMM128_FAR
                extra_slots = 2

            if slot + 1 + extra_slots > code_count:
                result["unwind_parse_warnings"].append(
                    f"{record['operation']} operand exceeds the declared UNWIND_CODE count."
                )
                result["unwind_operations"].append(record)
                break

            operand_offset = offset + 2
            operand_size = extra_slots * 2
            if extra_slots and operand_offset + operand_size > len(data):
                record["operand_truncated"] = True
                result["unwind_parse_warnings"].append(f"{record['operation']} operand slots are truncated.")
                result["unwind_operations"].append(record)
                break
            operand = int.from_bytes(data[operand_offset:operand_offset + operand_size], "little") if extra_slots else None
            if operation == 0:
                record["register"] = register_names.get(op_info, f"reg_{op_info}")
            elif operation == 1 and operand is not None:
                record["allocation_size"] = operand * 8 if op_info == 0 else operand
            elif operation == 2:
                record["allocation_size"] = op_info * 8 + 8
            elif operation == 3:
                record["frame_register"] = result["frame_register_name"]
                record["frame_offset"] = frame_scale * 16
            elif operation == 4 and operand is not None:
                record["register"] = register_names.get(op_info, f"reg_{op_info}")
                record["stack_offset"] = operand * 8
            elif operation == 5 and operand is not None:
                record["register"] = register_names.get(op_info, f"reg_{op_info}")
                record["stack_offset"] = operand
            elif operation == 8 and operand is not None:
                record["register"] = f"xmm{op_info}"
                record["stack_offset"] = operand * 16
            elif operation == 9 and operand is not None:
                record["register"] = f"xmm{op_info}"
                record["stack_offset"] = operand
            elif operation == 10:
                record["machine_frame_error_code"] = bool(op_info)
                record["machine_frame_size"] = 48 if op_info else 40
                if op_info > 1:
                    result["unwind_parse_warnings"].append("UWOP_PUSH_MACHFRAME has an invalid OpInfo.")
            elif operation == 6 and version != 2:
                result["unwind_parse_warnings"].append("UWOP_EPILOG is only defined for UNWIND_INFO version 2.")
            elif operation == 7:
                result["unwind_parse_warnings"].append("Reserved UWOP_SPARE_CODE was encountered.")
            elif operation > 10:
                result["unwind_parse_warnings"].append(f"Unknown unwind opcode {operation}.")
            if operand is not None:
                record["operand_raw"] = operand
            result["unwind_operations"].append(record)
            slot += 1 + extra_slots

        # The optional tail is aligned to a 4-byte boundary after the declared slots.
        tail_offset = 4 + ((code_count + 1) & ~1) * 2
        handler_flags = flags & 0x3
        if flags & 0x4 and handler_flags:
            result["unwind_parse_warnings"].append("CHAININFO cannot be combined with handler flags.")
        if flags & 0x4:
            if tail_offset + 12 <= len(data):
                begin_rva, end_rva, chained_unwind_rva = struct.unpack_from("<III", data, tail_offset)
                result["chained_begin_rva"] = begin_rva
                result["chained_end_rva"] = end_rva
                result["chained_unwind_info_rva"] = chained_unwind_rva
            else:
                result["unwind_parse_warnings"].append("Chained RUNTIME_FUNCTION record is truncated.")
        elif handler_flags:
            if tail_offset + 4 <= len(data):
                handler_rva = struct.unpack_from("<I", data, tail_offset)[0]
                if handler_rva:
                    result["handler_rva"] = handler_rva
                    result["language_data_offset"] = tail_offset + 4
                else:
                    result["unwind_parse_warnings"].append("Handler flags are set but the handler RVA is zero.")
            else:
                result["unwind_parse_warnings"].append("Exception/unwind handler RVA is truncated.")
        return result

    def section_for(self, va: int) -> Section | None:
        # Typical binaries have few sections; this is also used while decoding.
        for sec in self.sections:
            if sec.start <= va < sec.end:
                return sec
        return None

    def section_bytes(self, sec: Section) -> bytes:
        size = sec.backed_size
        if size <= 0 or sec.raw_offset < 0 or sec.raw_offset + size > len(self._data):
            return b""
        return self._data[sec.raw_offset:sec.raw_offset + size]

    def read_va(self, va: int, limit: int = 4096) -> bytes:
        sec = self.section_for(va)
        if sec is None:
            return b""
        delta = va - sec.start
        if delta < 0 or delta >= sec.backed_size:
            return b""
        n = min(limit, sec.backed_size - delta)
        off = sec.raw_offset + delta
        if off < 0 or off + n > len(self._data):
            return b""
        return self._data[off:off+n]

    def scan_pointer_slots(self, max_slots: int = 200_000) -> tuple[dict[int, int], dict[str, int | bool]]:
        """Recover bounded aligned pointers from readable, non-executable PE sections.

        A value is retained only when it points into a mapped image section. This
        supports static objects and vtables whose contents are represented by
        preferred-base VAs in the file; it does not apply relocations or prove that
        an object is live. Callers must validate code targets against decoded starts.
        """
        width = self.bits // 8
        fmt = "<Q" if self.bits == 64 else "<I"
        pointers: dict[int, int] = {}
        slots_scanned = 0
        capped = False
        for sec in self.sections:
            if not sec.readable or sec.executable:
                continue
            raw = self.section_bytes(sec)
            first = (-sec.start) % width
            for offset in range(first, len(raw) - width + 1, width):
                if slots_scanned >= max_slots:
                    capped = True
                    break
                slots_scanned += 1
                value = struct.unpack_from(fmt, raw, offset)[0]
                if self.section_for(value) is not None:
                    pointers[sec.start + offset] = value
            if capped:
                break
        return pointers, {
            "slots_scanned": slots_scanned,
            "pointer_slots_found": len(pointers),
            "slot_budget": max_slots,
            "budget_exhausted": capped,
        }

    def read_c_string(self, va: int, limit: int = 1024) -> str | None:
        raw = self.read_va(va, limit).split(b"\0", 1)[0]
        if len(raw) < 1:
            return None
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError:
            value = raw.decode("cp1252", errors="replace")
        if any(ord(c) < 32 and c not in "\r\n\t" for c in value):
            return None
        return value

    def scan_strings(self, min_length: int = 4, max_count: int = 200_000,
                     include_executable: bool = False) -> list[BinaryString]:
        """Scan file-backed data sections for ASCII and UTF-16LE strings.

        Executable sections are excluded by default to avoid reporting machine-code
        bytes as misleading strings. Enable include_executable only for special cases
        where a program intentionally embeds strings in executable code.
        """
        ascii_re = re.compile(rb"[\x20-\x7e]{" + str(min_length).encode() + rb",}")
        wide_re = re.compile(rb"(?:[\x20-\x7e]\x00){" + str(min_length).encode() + rb",}")
        found: dict[tuple[int, str], BinaryString] = {}
        for sec in self.sections:
            if sec.executable and not include_executable:
                continue
            blob = self.section_bytes(sec)
            if not blob:
                continue
            for match in ascii_re.finditer(blob):
                txt = match.group().decode("ascii", errors="replace")
                va = sec.start + match.start()
                found[(va, "ascii")] = BinaryString(va, txt, "ascii", sec.name, len(match.group()))
                if len(found) >= max_count: break
            if len(found) >= max_count: break
            for match in wide_re.finditer(blob):
                raw = match.group()
                try: txt = raw.decode("utf-16le")
                except UnicodeDecodeError: continue
                va = sec.start + match.start()
                found[(va, "utf-16le")] = BinaryString(va, txt, "utf-16le", sec.name, len(raw))
                if len(found) >= max_count: break
            if len(found) >= max_count: break
        return sorted(found.values(), key=lambda x: x.address)

    def close(self) -> None:
        try: self._data.close()
        finally:
            self._file.close()
            self.pe.close()
