"""Small, explicit signature hints for common Win32 and C runtime imports.

This catalog is intentionally incomplete. Unknown imports keep heuristic arity
and return typing rather than inheriting a guessed prototype.
"""

from __future__ import annotations

from typing import TypedDict


class ApiSignature(TypedDict):
    arity: int | None
    return_type: str | None


API_SIGNATURES: dict[str, ApiSignature] = {
    "getlasterror": {"arity": 0, "return_type": "uint32_t"},
    "abort": {"arity": 0, "return_type": "void"},
    "_cexit": {"arity": 0, "return_type": "void"},
    "exit": {"arity": 1, "return_type": "void"},
    "_exit": {"arity": 1, "return_type": "void"},
    "sleep": {"arity": 1, "return_type": "void"},
    "malloc": {"arity": 1, "return_type": "uintptr_t"},
    "calloc": {"arity": 2, "return_type": "uintptr_t"},
    "realloc": {"arity": 2, "return_type": "uintptr_t"},
    "free": {"arity": 1, "return_type": "void"},
    "strlen": {"arity": 1, "return_type": "size_t"},
    "puts": {"arity": 1, "return_type": "int32_t"},
    "fflush": {"arity": 1, "return_type": "int32_t"},
    "memcpy": {"arity": 3, "return_type": "uintptr_t"},
    "memmove": {"arity": 3, "return_type": "uintptr_t"},
    "memset": {"arity": 3, "return_type": "uintptr_t"},
    "virtualquery": {"arity": 3, "return_type": "size_t"},
    "virtualprotect": {"arity": 4, "return_type": "bool"},
    "setvbuf": {"arity": 3, "return_type": "int32_t"},
    "printf": {"arity": None, "return_type": "int32_t"},
    "fprintf": {"arity": None, "return_type": "int32_t"},
    "sprintf": {"arity": None, "return_type": "int32_t"},
    "snprintf": {"arity": None, "return_type": "int32_t"},
    "_aligned_malloc": {"arity": 2, "return_type": "uintptr_t"},
    "aligned_alloc": {"arity": 2, "return_type": "uintptr_t"},
    "_aligned_free": {"arity": 1, "return_type": "void"},
    "strdup": {"arity": 1, "return_type": "uintptr_t"},
    "_strdup": {"arity": 1, "return_type": "uintptr_t"},
    "wcsdup": {"arity": 1, "return_type": "uintptr_t"},
    "heapalloc": {"arity": 3, "return_type": "uintptr_t"},
    "heaprealloc": {"arity": 4, "return_type": "uintptr_t"},
    "heapfree": {"arity": 3, "return_type": "bool"},
    "virtualalloc": {"arity": 4, "return_type": "uintptr_t"},
    "virtualallocex": {"arity": 5, "return_type": "uintptr_t"},
    "localalloc": {"arity": 2, "return_type": "uintptr_t"},
    "globalalloc": {"arity": 2, "return_type": "uintptr_t"},
    "localfree": {"arity": 1, "return_type": "uintptr_t"},
    "globalfree": {"arity": 1, "return_type": "uintptr_t"},
    "cotaskmemalloc": {"arity": 1, "return_type": "uintptr_t"},
    "cotaskmemrealloc": {"arity": 2, "return_type": "uintptr_t"},
    "cotaskmemfree": {"arity": 1, "return_type": "void"},
    "sysallocstring": {"arity": 1, "return_type": "uintptr_t"},
    "sysfreestring": {"arity": 1, "return_type": "void"},
    "strcpy": {"arity": 2, "return_type": "uintptr_t"},
    "strncpy": {"arity": 3, "return_type": "uintptr_t"},
    "strcat": {"arity": 2, "return_type": "uintptr_t"},
    "getprocaddress": {"arity": 2, "return_type": "uintptr_t"},
    "getmodulehandlea": {"arity": 1, "return_type": "uintptr_t"},
    "getmodulehandlew": {"arity": 1, "return_type": "uintptr_t"},
    "loadlibrarya": {"arity": 1, "return_type": "uintptr_t"},
    "loadlibraryw": {"arity": 1, "return_type": "uintptr_t"},
    "getcurrentprocess": {"arity": 0, "return_type": "uintptr_t"},
    "getcurrentthread": {"arity": 0, "return_type": "uintptr_t"},
    "getstdhandle": {"arity": 1, "return_type": "uintptr_t"},
    "openprocess": {"arity": 3, "return_type": "uintptr_t"},
    "openthread": {"arity": 3, "return_type": "uintptr_t"},
    "createfilea": {"arity": 7, "return_type": "uintptr_t"},
    "createfilew": {"arity": 7, "return_type": "uintptr_t"},
    "closehandle": {"arity": 1, "return_type": "bool"},
    "createeventa": {"arity": 4, "return_type": "uintptr_t"},
    "createeventw": {"arity": 4, "return_type": "uintptr_t"},
    "createmutexa": {"arity": 3, "return_type": "uintptr_t"},
    "createmutexw": {"arity": 3, "return_type": "uintptr_t"},
    "findfirstfilea": {"arity": 2, "return_type": "uintptr_t"},
    "findfirstfilew": {"arity": 2, "return_type": "uintptr_t"},
    "findnextfilea": {"arity": 2, "return_type": "bool"},
    "findnextfilew": {"arity": 2, "return_type": "bool"},
    "isdebuggerpresent": {"arity": 0, "return_type": "bool"},
}

# Parameter types are sparse: only prototypes useful for pointer and handle
# flow are listed. Integer positions not present here remain untyped.
API_PARAMETER_TYPES: dict[str, tuple[str, ...]] = {
    "malloc": ("size_t",),
    "calloc": ("size_t", "size_t"),
    "realloc": ("uintptr_t", "size_t"),
    "free": ("uintptr_t",),
    "strlen": ("const char *",),
    "memcpy": ("void *", "const void *", "size_t"),
    "memmove": ("void *", "const void *", "size_t"),
    "memset": ("void *", "int32_t", "size_t"),
    "getprocaddress": ("uintptr_t", "const char *"),
    "getmodulehandlea": ("const char *",),
    "getmodulehandlew": ("const wchar_t *",),
    "loadlibrarya": ("const char *",),
    "loadlibraryw": ("const wchar_t *",),
    "getstdhandle": ("int32_t",),
    "openprocess": ("uint32_t", "bool", "uint32_t"),
    "openthread": ("uint32_t", "bool", "uint32_t"),
    "createfilea": ("const char *", "uint32_t", "uint32_t", "void *", "uint32_t", "uint32_t", "uintptr_t"),
    "createfilew": ("const wchar_t *", "uint32_t", "uint32_t", "void *", "uint32_t", "uint32_t", "uintptr_t"),
    "closehandle": ("uintptr_t",),
    "createeventa": ("void *", "bool", "bool", "const char *"),
    "createeventw": ("void *", "bool", "bool", "const wchar_t *"),
    "createmutexa": ("void *", "bool", "const char *"),
    "createmutexw": ("void *", "bool", "const wchar_t *"),
    "findfirstfilea": ("const char *", "void *"),
    "findfirstfilew": ("const wchar_t *", "void *"),
    "findnextfilea": ("uintptr_t", "void *"),
    "findnextfilew": ("uintptr_t", "void *"),
    "virtualalloc": ("void *", "size_t", "uint32_t", "uint32_t"),
    "virtualallocex": ("uintptr_t", "void *", "size_t", "uint32_t", "uint32_t"),
    "virtualprotect": ("void *", "size_t", "uint32_t", "uint32_t *"),
}


def normalize_api_name(name: str | None) -> str:
    if not name:
        return ""
    value = name.split("!")[-1].strip().lower()
    value = value.split("@", 1)[0]
    for prefix in ("__imp_", "_imp__", "imp_"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    return value


def get_api_signature(name: str | None) -> ApiSignature | None:
    return API_SIGNATURES.get(normalize_api_name(name))
