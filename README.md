# Independent Decompiler Engine 0.2.0

A modular **independent static analysis and conservative pseudocode research engine** for Windows PE x86/x86-64 binaries. It does not call Ghidra or use it as a backend.

## What's new in 0.2.0

The main new capability is bounded, function-local **forward symbolic data-flow analysis**. It propagates simple constants and expressions through register aliases and stack-relative slots, follows CFG edges, merges facts conservatively at control-flow joins, tracks compare/test conditions to conditional branches, and records direct-call result expressions using ABI-argument-count heuristics. The output is `value_flow.jsonl`, with run-level statistics in `dataflow_summary.json`.

The data-flow pass is deliberately a foundation for higher-level reconstruction—not a claim that arbitrary compiled programs can now be faithfully converted back into their original source. Values become unknown when paths disagree, calls have unresolved arguments, or expressions become too complex. Loops are bounded by an iteration/step cap. Original identifiers, complete types, exact source structure, and general loop/if reconstruction are not yet recovered.

Constant integer folding respects the destination width for common arithmetic,
bitwise, shift, and unary operations. Shift counts use the x86 five-bit or
six-bit mask, as appropriate. Carry tracking supports a bounded CF subset for
`cmp`/`test`, add/subtract, `clc`/`stc`, and known-carry `adc`/`sbb`; common
condition flags also support equality, signed/unsigned, overflow, sign, and
parity branches when their inputs are known. Common shifts track defined CF/OF
cases and preserve flags for a zero effective count; undefined shift flags,
multiply flags, and full CPU flag interactions remain unknown.

## Current capabilities

- Parses Windows PE headers, executable/data sections, imports and exports.
- Memory-maps the input for one-pass section access.
- Linear-sweep Capstone decoding for x86/x86-64 executable sections.
- Builds basic blocks and direct branch edges.
- Recovers bounded absolute, image-relative or x64 signed-relative executable
  targets from simple indirect jump tables and adds them to the CFG when safely
  resolvable.
- Records jump-table entry size/base, contiguous valid entries observed, and proven
  indirect-call targets in instruction metadata. Observed table entries are evidence,
  not a proven switch-table length.
- Models common bit-test, sign-extension and flag-preserving unary operations
  explicitly in normalized IR.
- Candidate function starts from PE entry, exports, Windows x64 `.pdata` runtime-function records and direct calls.
- Decodes bounded Windows x64 unwind-code operations, handler RVAs and chained
  runtime-function records; decoded executable handler addresses become candidates.
- Recognizes isolated one-instruction direct-jump thunks to known functions when
  bounded root reachability shows the block is outside known functions.
- Records direct jumps to known function entries as tail-call/forwarding edges in
  each function's call references.
- ASCII/UTF-16LE string scan excludes executable sections by default.
- Emits function references, imports, string references, CFG and register-level pseudocode.
- Adds value flow tracking for constants, register aliases, stack-relative values, calls and branch conditions.
- Tracks stack and non-stack memory access widths, invalidates overlapping stack
  facts, and avoids reusing values for width-mismatched reads.
- Canonicalizes simple memory accesses through equal known pointer values and
  constant offsets, including register copies, pointer increments, and one stable
  scaled array index.
- Resolves indirect call operands in value-flow reports when a propagated concrete
  address matches a discovered function entry; calls without concrete targets remain unresolved.
- Resolves a function pointer loaded from a known IAT slot and later called through
  a register, preserving its import symbol in value-flow and pseudocode reports.
- Scans a bounded number of aligned pointers in readable non-executable PE sections;
  a recovered table base plus slot can resolve a memory-indirect call when the slot
  points to a decoded code address.
- Infers case counts for common scaled-index jump tables guarded by matching unsigned
  `cmp` plus `ja`/`jae` checks, reports the check as evidence, and caps table scanning
  at the inferred count.
- Recovers heuristic x86 stack parameters from conventional EBP frames and from
  ESP-relative reads when entry-block stack depth is locally trackable. It also
  normalizes observed Windows x64 stack parameters beyond shadow space through simple
  entry-block RSP adjustments and established RBP frames. It uses `ret N` cleanup
  and ECX/EDX evidence to classify x86 convention candidates. Direct calls use known
  signatures, x86 right-to-left `push` sequences, and tracked x64 outgoing RSP slots
  (including right-to-left pushes followed by shadow-space allocation). Stack depth,
  constants, register copies, unchanged argument values, and simple immediate
  arithmetic can flow across CFG edges when all paths agree at each join. General
  memory-derived and vector/FPU values remain unresolved. Known x86 callee signatures
  retain placeholders for missing pushed arguments, and simple caller stack cleanup
  removes consumed push values before later calls.
- Carries conservative types through data-flow states, SSA versions and phi
  nodes, widening conflicting join types to `unknown`.
- Propagates recovered direct-callee return types into call-result data flow and
  canonicalizes `rbp`/`ebp` and `rsp`/`esp` stack aliases.
- Names unresolved positive frame/stack accesses as `stack_arg_<offset>` and
  tracks repeated non-stack memory expressions within one abstract path, with
  equivalent numeric offset spellings canonicalized while preserving register widths.
- Invalidates non-stack memory facts across calls and ambiguous non-stack writes
  instead of reusing potentially stale values.
- Structures simple, unambiguous two-way CFG diamonds as C-like `if`/`else` blocks.
- Structures simple natural loops as C-like `while` blocks when the loop header,
  body back-edge and exit are unambiguous.
- Structures acyclic multi-block loop bodies with a single header and exit,
  converting explicit edges to the header and exit into `continue` and `break`.
  Internal branches remain labels and `goto` statements.
- Structures simple single-block post-tested loops as C-like `do`/`while` blocks.
- Structures small direct equality-test dispatch chains as C-like `switch` blocks.
- Preserves safe direct case-to-case switch fallthrough without inserting an
  incorrect `break`.
- Repeats safe structuring passes so sequential and mixed recognized regions
  can be reduced within one function.
- Bounded output budgets keep large inputs from producing unbounded analysis reports.
- No external decompiler is invoked.

## Output files

- `summary.json`: architecture, section inventory, analysis counts, warnings and stage timings.
- `functions.json`: function-entry candidates and call/string references.
- `cfg.json`: basic blocks and successors.
- `strings.json`: strings in file-backed non-executable sections.
- `crypto_findings.json`: algorithm use states (`string_only`, `symbol_present`, `call_site_found`, `statically_reachable`, and `unknown`) with evidence, confidence, Go build metadata, function symbols, and direct or statically resolved indirect calls. Unresolved indirect calls are recorded without guessed edges. Go pclntab recovery currently covers Go 1.18+ x64 function names; inline trees are not decoded. `runtime_observed` is false because analysis is static.
- `crypto_cbom.cyclonedx.json`: CycloneDX 1.7 CBOM export aggregating algorithm-family variants while retaining evidence locations, analyzer version, executable hash, callers/callees, Go module metadata, static states, and analysis limitations. Exclude generated reports and scanner source/test files when scanning a project with another CBOM scanner to avoid recursively counting the scanner itself.
- `tests/fixtures/crypto_accuracy/`: known Go binary fixtures for AES, RSA, string-only, unreachable crypto, and indirect crypto. The end-to-end fixture suite needs Go, `pefile`, and Capstone; it skips when those dependencies are unavailable.
- `imports.json`: imported symbols and IAT addresses.
- `xrefs.csv`: references to known imports and string addresses.
- `pseudocode.c.txt`: conservative register-level C-like pseudocode; not guaranteed compilable C.
- `program.json`: documented Ghidra-inspired program model containing PE memory blocks,
  imports/exports, discovered functions, references and analysis warnings. This is an
  interoperability report, not a Ghidra project/database file. It also records resolved
  indirect calls and candidate indirect jump-table evidence, including contiguous entries
  observed without claiming a proven table bound, plus decoded Windows x64 unwind metadata.
- `decompiler.jsonl`: one JSON record per emitted function containing its address, name,
  recovered signature, CFG blocks and the same decompiler text written to `pseudocode.c.txt`.
- `value_flow.jsonl`: per-instruction resolved operand expressions, writes, inferred branch conditions, and register/stack state after the instruction.
- `dataflow_summary.json`: count of functions/rows, limits, per-function coverage and warnings.
- `assembly.asm` and `ir.jsonl`: optional verbose diagnostics, not written with `--fast-output`.
- `normalized_ir.jsonl`, `ssa_ir.jsonl`, `ssa_cfg.json`, and `ssa_summary.json`: normalized operation and SSA analysis reports. These are produced even with `--fast-output`; at present, SSA is not yet used to recover source-level locals in `pseudocode.c.txt`.

## Windows setup

Use Python 3.10+ and PowerShell from this directory:

```powershell
py -m pip install -r requirements.txt
```

Or run `install_windows.ps1` (PowerShell execution policy might need process-scoped permission).

Analyze a file:

```powershell
py .\independent_decompiler_v6.py "C:\path\to\program.exe" --out "analysis_output" --fast-output
```

Default data-flow limits are 200,000 report rows and 500 candidate functions. Adjust them if needed:

```powershell
py .\independent_decompiler_v6.py "C:\path\to\program.exe" --out "analysis_large" --fast-output --max-dataflow-instructions 200000 --max-dataflow-functions 500
```

Disable the value-flow pass when you need only the faster inventory/CFG reports:

```powershell
py .\independent_decompiler_v6.py "C:\path\to\program.exe" --out "analysis_fast" --fast-output --no-dataflow
```

## Validation

Run the test suite:

```powershell
py -m pytest -q
```

The tests cover control-flow edges, basic instruction lifting, constant propagation into stack slots, arithmetic folding, branch-condition propagation and conservative state merging. The test suite does not substitute for an end-to-end regression run against a real PE binary. This environment did not have `pefile`/`capstone` installed for a full binary-analysis run, so run it on Windows and inspect `summary.json` and `dataflow_summary.json`.

## Known limitations

- Linear sweep can mistake embedded data for instructions; skip-data bytes are placeholders.
- Indirect calls, malformed or complex jump tables, exception-driven control flow,
  packed/obfuscated code, overlapping code and runtime-generated code remain difficult.
- Proven indirect calls through an exported code address and bounded value-flow
  resolution of register-held targets that match decoded instruction starts are
  added as function candidates. The value-flow discovery pass has its own row cap
  equal to the configured data-flow budget. It can misidentify linear-sweep
  decodings of embedded data; arbitrary memory-held function pointers remain
  unresolved. Isolated jump-thunk recognition is conservative and disabled when
  a root function scan reaches its block cap.
- Runtime metadata provides useful boundaries but can include CRT/runtime functions and is not synonymous with application functions.
- Function arity and calling conventions are heuristic. ESP argument recovery only
  tracks straight-line entry-block pushes, pops, and constant ESP adjustments; it
  stops at calls, branches, or alignment masks. x64 outgoing stack arguments require
  a known callee and tracked same-block RSP slot stores. Ambiguous thiscall/fastcall
  cases and complete caller/callee cleanup behavior remain open.
- The value-flow analysis is bounded, conservative and intraprocedural; it is not full x86 semantics or whole-program SSA.
- Normalized IR type hints are conservative (`uintN_t`, `uintptr_t`, `bool`,
  `flags_t` and stack-value categories); they are not recovered source-level
  declarations or full pointer/structure types.
- Pseudocode remains machine-/register-level in many functions. Only simple two-way
  conditional diamonds that reconverge at one block are currently structured as
  `if`/`else`, bounded acyclic single-exit natural loops are structured as
  `while` (including multi-block bodies with explicit `break`/`continue` edges),
  and single-block post-tested loops are structured as `do`/`while`;
  small direct equality-test dispatch chains are structured as `switch`; indirect
  jump tables, irreducible graphs and ambiguous flow remain labels and `goto`
  statements. Repeated passes support safe mixed regions, while complex or
  ambiguous switch fallthrough and jump tables remain conservative. It does
  not reliably reconstruct original source code for arbitrary executables.

Stack-passed arguments are reported symbolically when positive frame/stack
offsets are visible, but complete ABI layouts across prologues, tail calls and
variadic functions are not proven. Indirect calls without a direct import or
proven target remain explicitly unresolved; exact memory-expression tracking is
not a general alias analysis.

The remaining research work is broader function-boundary recovery, exception and
unwind-flow analysis, complete memory aliasing, and source-level type/structure
recovery. These require larger architectural phases rather than isolated output
patterns.

See [PENDING_WORK_STATUS.md](PENDING_WORK_STATUS.md) for a status audit of the full
backlog and the remaining implementation phases.
#   b i n a r y - a n a l y s i s  
 