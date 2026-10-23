from __future__ import annotations

import argparse
import signal
import sys
from threading import Event
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent PE decompiler research engine (no Ghidra runtime or launcher)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input Windows PE .exe or .dll")
    parser.add_argument("--out", "-o", default="analysis_output", help="Output folder")
    parser.add_argument("--max-instructions", type=int, default=2_000_000)
    parser.add_argument("--max-functions", type=int, default=2_000)
    parser.add_argument("--max-pseudocode-functions", type=int, default=500)
    parser.add_argument("--max-blocks-per-function", type=int, default=1_000)
    parser.add_argument("--max-pseudocode-instructions", type=int, default=200_000)
    parser.add_argument("--min-string-length", type=int, default=4)
    parser.add_argument("--fast-output", action="store_true", help="Skip full assembly and per-instruction IR outputs; keep normalized IR and SSA analysis")
    parser.add_argument("--no-assembly", action="store_true")
    parser.add_argument("--no-ir", action="store_true")
    parser.add_argument("--no-dataflow", action="store_true")
    parser.add_argument("--max-dataflow-instructions", type=int, default=200_000)
    parser.add_argument("--max-dataflow-functions", type=int, default=500)
    parser.add_argument("--progress", action="store_true", help="Report stage and per-function progress to stderr")
    parser.add_argument("--version", action="version", version="Independent Decompiler Engine 0.4.0-dev (independent IR + SSA; no Ghidra dependency)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: input file does not exist: {input_path}", file=sys.stderr)
        return 2
    positive_limits = (
        args.max_instructions, args.max_functions, args.max_pseudocode_functions,
        args.max_blocks_per_function, args.max_pseudocode_instructions,
        args.min_string_length, args.max_dataflow_instructions, args.max_dataflow_functions,
    )
    if any(value < 1 for value in positive_limits):
        print("ERROR: limits must be positive integers", file=sys.stderr)
        return 2
    try:
        from decompiler_engine.pipeline import DecompilerPipeline
    except ModuleNotFoundError as exc:
        if exc.name in {"pefile", "capstone"}:
            print("Missing dependency. Run: py -m pip install -r requirements.txt", file=sys.stderr)
            return 2
        raise
    cancel_event = Event()

    def progress(event: dict[str, object]) -> None:
        if args.progress:
            total = int(event.get("total", 0))
            completed = int(event.get("completed", 0))
            stage = str(event.get("stage", "analysis"))
            print(f"\r{stage}: {completed}/{total}", end="", file=sys.stderr, flush=True)

    def request_cancel(_signum, _frame) -> None:
        cancel_event.set()
        print("\nCancellation requested; stopping at the next function boundary.", file=sys.stderr)

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, request_cancel)
    try:
        pipeline = DecompilerPipeline(
            str(input_path), args.out,
            max_instructions=args.max_instructions,
            max_functions=args.max_functions,
            max_pseudocode_functions=args.max_pseudocode_functions,
            max_blocks_per_function=args.max_blocks_per_function,
            max_pseudocode_instructions=args.max_pseudocode_instructions,
            min_string_length=args.min_string_length,
            write_assembly=not (args.fast_output or args.no_assembly),
            write_ir=not (args.fast_output or args.no_ir),
            write_ssa=True,
            write_dataflow=not args.no_dataflow,
            max_dataflow_instructions=args.max_dataflow_instructions,
            max_dataflow_functions=args.max_dataflow_functions,
            progress_callback=progress,
            cancel_check=cancel_event.is_set,
        )
        print(f"Input: {input_path.resolve()}")
        print("Running independent PE parsing, disassembly, CFG, normalized IR, SSA, data-flow and pseudocode reconstruction...")
        summary = pipeline.run()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        if args.progress:
            print(file=sys.stderr)
    print(f"Architecture: {summary['architecture']}")
    print(f"Decoded instructions: {summary['instruction_count']:,}")
    print(f"Basic blocks: {summary['basic_block_count']:,}")
    print(f"Function candidates: {summary['function_candidates']:,}")
    print(f"Strings: {summary['string_count']:,}; imports: {summary['import_count']:,}")
    print(f"Data-flow rows: {summary.get('dataflow_rows', 0):,}; functions analyzed: {summary.get('dataflow_functions_analyzed', 0):,}")
    print(f"Normalized IR operations: {summary.get('normalized_ir_operations', 0):,}")
    print(f"SSA operations: {summary.get('ssa_operations', 0):,}; phi nodes: {summary.get('ssa_phi_nodes', 0):,}; functions analyzed: {summary.get('ssa_functions_analyzed', 0):,}")
    print(f"Total time: {summary['timings_seconds']['total']:.3f}s")
    print(f"Output: {Path(args.out).resolve()}")
    for warning in summary.get("warnings", []):
        print("WARNING:", warning)
    return 130 if summary.get("cancelled") else 0


if __name__ == "__main__":
    raise SystemExit(main())
