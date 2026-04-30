"""
36-cross-section orchestrator for the Forest Through the Trees project.

For each of the 36 (Size × X × Y) triples in Table D.1:
  1. Set AP_TREES_CHARS env var to the triple's panel column names
  2. Run ap-trees/run_all.py        -> ap-trees/outputs/<TRIPLE>/backtest_comparison.csv
  3. Run TripleSort/run_all.py      -> TripleSort/outputs/<TRIPLE>/backtest_comparison.csv
  4. Concatenate the two CSVs into  ./backtest_results/<triple_lower>/backtest_comparison.csv
     (the location your aggregate_table_d1.py reads from)

Resume support: any triple whose merged backtest_comparison.csv already exists
is skipped. To force a rerun, delete the per-triple output dir.

Logging: each subprocess's stdout/stderr is teed to a per-triple log file.

Usage
-----
    # Run all 36 (skip any already-done):
    python orchestrate_36.py

    # Run a specific subset by id:
    python orchestrate_36.py --only 16 1 2

    # Run all, but force rerun even if outputs exist:
    python orchestrate_36.py --force

    # Run with a different repo root:
    python orchestrate_36.py --repo-root /path/to/forest-through-the-trees

    # Skip the AP-Trees runs (e.g., to re-do just TripleSort):
    python orchestrate_36.py --skip-aptrees

    # Dry run -- print what would happen without doing it:
    python orchestrate_36.py --dry-run

Failure semantics
-----------------
If any subprocess returns non-zero exit code, that triple is marked as failed
and the orchestrator continues to the next. A summary at the end lists
successes and failures. Re-running picks up the failed ones (since their
output CSVs don't exist).

Long-running advice
-------------------
This script can take ~36 hours for all 36 triples on a typical laptop. To run
in the background and survive terminal disconnects:

    nohup python orchestrate_36.py > orchestrator.log 2>&1 &
    echo $! > orchestrator.pid

Then check progress with:

    tail -f orchestrator.log
    tail -f logs/<triple>/aptrees.log

Or use tmux/screen:

    tmux new -s figure7
    python orchestrate_36.py
    # detach with Ctrl-b d, reattach with: tmux attach -t figure7
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Pull the same triples the figure pipeline knows about. The orchestrator
# lives in cross-section-regression-plots; cross_sections.py is a sibling.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sections import CROSS_SECTIONS, CHAR_NAME_MAP, CrossSection


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    repo_root: Path
    aptrees_dir_name: str = "ap-trees"
    triplesort_dir_name: str = "TripleSort"
    aptrees_entry: str = "run_all.py"
    triplesort_entry: str = "run_all.py"
    # Output-merging destination root. The figure aggregator reads from here.
    merge_out_dir: Path = field(default=None)
    # Logs root.
    log_dir: Path = field(default=None)
    # Which Python to use for subprocesses. None means "the same one running
    # this script". Override with --python /path/to/venv/bin/python.
    python_exe: str = sys.executable

    def __post_init__(self):
        self.repo_root = self.repo_root.resolve()
        if self.merge_out_dir is None:
            # Default to figures/Code/backtest_results, where aggregate_table_d1
            # currently looks. Change this if you've moved the figures pipeline.
            self.merge_out_dir = self.repo_root / "figures" / "Code" / "backtest_results"
        if self.log_dir is None:
            self.log_dir = self.repo_root / "orchestrator_logs"
        self.merge_out_dir = Path(self.merge_out_dir)
        self.log_dir = Path(self.log_dir)


# ---------------------------------------------------------------------------
# Per-triple work
# ---------------------------------------------------------------------------

@dataclass
class TripleResult:
    cs: CrossSection
    aptrees_status: str = "pending"     # "ok", "skipped", "failed", "pending"
    triplesort_status: str = "pending"
    merge_status: str = "pending"
    elapsed_sec: float = 0.0
    error: Optional[str] = None


def _triple_dir_name(cs: CrossSection) -> str:
    """Match the existing convention: 'LME_OP_Investment' (uppercase)."""
    return "_".join(CHAR_NAME_MAP[c] for c in (cs.char1, cs.char2, cs.char3))


def _triple_chars_csv(cs: CrossSection) -> str:
    """Format for the AP_TREES_CHARS env var: 'LME,OP,Investment'."""
    return ",".join(CHAR_NAME_MAP[c] for c in (cs.char1, cs.char2, cs.char3))


def _project_output_path(project_dir: Path, cs: CrossSection) -> Path:
    """Where a backtest project writes its backtest_comparison.csv for this CS."""
    return project_dir / "outputs" / _triple_dir_name(cs) / "backtest_comparison.csv"


def _merged_output_path(cfg: Config, cs: CrossSection) -> Path:
    """Where we write the merged backtest_comparison.csv that the figure pipe reads."""
    # The figure pipeline uses the lowercase key ('lme_op_investment').
    return cfg.merge_out_dir / cs.key / "backtest_comparison.csv"


def run_subprocess(
    cmd: list[str],
    cwd: Path,
    env: dict,
    log_path: Path,
    description: str,
) -> tuple[int, float]:
    """Run a subprocess, tee output to log_path, return (returncode, elapsed_sec)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [{description}] starting -> {log_path}", flush=True)
    start = time.monotonic()

    with open(log_path, "w") as logf:
        # Stamp the log so future-you knows when this ran.
        logf.write(f"# {description}\n")
        logf.write(f"# cwd: {cwd}\n")
        logf.write(f"# cmd: {' '.join(cmd)}\n")
        logf.write(f"# AP_TREES_CHARS: {env.get('AP_TREES_CHARS', '<unset>')}\n")
        logf.write(f"# TS_STATIC_ONLY: {env.get('TS_STATIC_ONLY', '<unset>')}\n")
        logf.write(f"# started: {dt.datetime.now().isoformat()}\n")
        logf.write("# " + "=" * 70 + "\n")
        logf.flush()

        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
        rc = proc.wait()

    elapsed = time.monotonic() - start
    print(f"  [{description}] returncode={rc}  elapsed={elapsed/60:.1f} min", flush=True)
    return rc, elapsed


def merge_backtest_csvs(
    aptrees_csv: Path,
    triplesort_csv: Path,
    merged_csv: Path,
) -> None:
    """Concatenate the two backtest_comparison.csv files into one.

    Both files have the same schema:
        date, date_dt, yy, mm, method, gross_ret, turnover_raw, turnover, cost, net_ret

    We keep all rows from both (no dedup). The `method` column distinguishes
    AP-Trees variants (A1, A2, B, C) from TripleSort variants. If there's any
    overlap on (method, date_dt) we keep the AP-Trees row and warn — that
    would indicate a project naming collision.
    """
    if not aptrees_csv.exists():
        raise FileNotFoundError(f"AP-Trees output missing: {aptrees_csv}")
    if not triplesort_csv.exists():
        raise FileNotFoundError(f"TripleSort output missing: {triplesort_csv}")

    a = pd.read_csv(aptrees_csv)
    t = pd.read_csv(triplesort_csv)

    a_methods = set(a["method"].unique())
    t_methods = set(t["method"].unique())
    overlap = a_methods & t_methods
    if overlap:
        print(f"  WARNING: method-name collision between projects: {overlap}. "
              f"Keeping AP-Trees rows, dropping TripleSort rows for these methods.")
        t = t[~t["method"].isin(overlap)]

    merged = pd.concat([a, t], ignore_index=True)
    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_csv, index=False)
    print(f"  merged -> {merged_csv}  ({len(a)} AP-Tree + {len(t)} TS = {len(merged)} rows)")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_one(
    cs: CrossSection,
    cfg: Config,
    skip_aptrees: bool,
    skip_triplesort: bool,
    ts_static_only: bool,
    force: bool,
    dry_run: bool,
) -> TripleResult:
    result = TripleResult(cs=cs)
    triple_dir_name = _triple_dir_name(cs)
    chars_csv = _triple_chars_csv(cs)

    aptrees_dir = cfg.repo_root / cfg.aptrees_dir_name
    triplesort_dir = cfg.repo_root / cfg.triplesort_dir_name
    aptrees_csv = _project_output_path(aptrees_dir, cs)
    triplesort_csv = _project_output_path(triplesort_dir, cs)
    merged_csv = _merged_output_path(cfg, cs)
    log_root = cfg.log_dir / triple_dir_name

    print(f"\n{'='*72}")
    print(f"[{cs.id:2d}] {triple_dir_name}  (AP_TREES_CHARS={chars_csv})")
    print(f"     merged target: {merged_csv}")

    # Resume: skip if final merged output already exists and we're not forcing.
    if merged_csv.exists() and not force:
        print("  -> SKIP: merged output already exists. Use --force to rerun.")
        result.aptrees_status = "skipped"
        result.triplesort_status = "skipped"
        result.merge_status = "skipped"
        return result

    if dry_run:
        print(f"  [DRY RUN] would set AP_TREES_CHARS={chars_csv}")
        if not skip_aptrees:
            print(f"  [DRY RUN] would run: cd {aptrees_dir} && python {cfg.aptrees_entry}")
        if not skip_triplesort:
            print(f"  [DRY RUN] would run: cd {triplesort_dir} && python {cfg.triplesort_entry}")
        print(f"  [DRY RUN] would merge -> {merged_csv}")
        result.aptrees_status = result.triplesort_status = "dry-run"
        result.merge_status = "dry-run"
        return result

    env = os.environ.copy()
    env["AP_TREES_CHARS"] = chars_csv
    if ts_static_only:
        env["TS_STATIC_ONLY"] = "1"
    triple_start = time.monotonic()

    # AP-Trees
    if skip_aptrees:
        result.aptrees_status = "skipped (--skip-aptrees)"
    elif aptrees_csv.exists() and not force:
        print(f"  AP-Trees output exists -> skipping subprocess: {aptrees_csv}")
        result.aptrees_status = "skipped (output exists)"
    else:
        rc, _ = run_subprocess(
            cmd=[cfg.python_exe, cfg.aptrees_entry],
            cwd=aptrees_dir,
            env=env,
            log_path=log_root / "aptrees.log",
            description=f"AP-Trees cs#{cs.id}",
        )
        if rc != 0:
            result.aptrees_status = f"failed (rc={rc})"
            result.error = f"AP-Trees subprocess returned {rc}"
            result.elapsed_sec = time.monotonic() - triple_start
            return result
        if not aptrees_csv.exists():
            result.aptrees_status = "failed (no output)"
            result.error = f"AP-Trees subprocess succeeded but didn't write {aptrees_csv}"
            result.elapsed_sec = time.monotonic() - triple_start
            return result
        result.aptrees_status = "ok"

    # TripleSort
    if skip_triplesort:
        result.triplesort_status = "skipped (--skip-triplesort)"
    elif triplesort_csv.exists() and not force:
        print(f"  TripleSort output exists -> skipping subprocess: {triplesort_csv}")
        result.triplesort_status = "skipped (output exists)"
    else:
        rc, _ = run_subprocess(
            cmd=[cfg.python_exe, cfg.triplesort_entry],
            cwd=triplesort_dir,
            env=env,
            log_path=log_root / "triplesort.log",
            description=f"TripleSort cs#{cs.id}",
        )
        if rc != 0:
            result.triplesort_status = f"failed (rc={rc})"
            result.error = f"TripleSort subprocess returned {rc}"
            result.elapsed_sec = time.monotonic() - triple_start
            return result
        if not triplesort_csv.exists():
            result.triplesort_status = "failed (no output)"
            result.error = f"TripleSort subprocess succeeded but didn't write {triplesort_csv}"
            result.elapsed_sec = time.monotonic() - triple_start
            return result
        result.triplesort_status = "ok"

    # Merge step
    try:
        merge_backtest_csvs(aptrees_csv, triplesort_csv, merged_csv)
        result.merge_status = "ok"
    except Exception as e:
        result.merge_status = "failed"
        result.error = f"Merge failed: {e}"

    result.elapsed_sec = time.monotonic() - triple_start
    print(f"  -> elapsed: {result.elapsed_sec/60:.1f} min")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path.cwd(),
                   help="Repository root containing ap-trees/ and TripleSort/. "
                        "Default: current directory.")
    p.add_argument("--only", type=int, nargs="*", default=None,
                   help="Cross-section IDs to run (1-36). Default: all 36.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if merged output exists. Be careful — this "
                        "will overwrite previous outputs.")
    p.add_argument("--skip-aptrees", action="store_true",
                   help="Skip AP-Trees subprocesses (only do TripleSort + merge).")
    p.add_argument("--skip-triplesort", action="store_true",
                   help="Skip TripleSort subprocesses (only do AP-Trees + merge).")
    p.add_argument("--ts-static-only", action="store_true",
                   help="Set TS_STATIC_ONLY=1 for TripleSort subprocesses.")
    p.add_argument("--python", default=sys.executable,
                   help=f"Python executable for subprocesses. Default: {sys.executable}")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without doing it.")
    args = p.parse_args()

    cfg = Config(repo_root=args.repo_root, python_exe=args.python)

    # Sanity-check the working tree before doing anything.
    aptrees_entry = cfg.repo_root / cfg.aptrees_dir_name / cfg.aptrees_entry
    triplesort_entry = cfg.repo_root / cfg.triplesort_dir_name / cfg.triplesort_entry
    if not aptrees_entry.exists():
        sys.exit(f"ERROR: AP-Trees entry point not found: {aptrees_entry}")
    if not triplesort_entry.exists():
        sys.exit(f"ERROR: TripleSort entry point not found: {triplesort_entry}")

    triples = CROSS_SECTIONS
    if args.only:
        triples = [cs for cs in CROSS_SECTIONS if cs.id in args.only]
        if not triples:
            sys.exit(f"--only IDs {args.only} matched no cross-sections")

    print(f"Repo root:       {cfg.repo_root}")
    print(f"AP-Trees:        {aptrees_entry}")
    print(f"TripleSort:      {triplesort_entry}")
    print(f"Merge target:    {cfg.merge_out_dir}")
    print(f"Logs:            {cfg.log_dir}")
    print(f"Python:          {cfg.python_exe}")
    print(f"Triples to run:  {len(triples)} ({[cs.id for cs in triples]})")
    print(f"Force:           {args.force}")
    print(f"TS static only:  {args.ts_static_only}")
    print(f"Dry run:         {args.dry_run}")

    overall_start = time.monotonic()
    results: list[TripleResult] = []
    for i, cs in enumerate(triples, 1):
        try:
            res = process_one(
                cs=cs,
                cfg=cfg,
                skip_aptrees=args.skip_aptrees,
                skip_triplesort=args.skip_triplesort,
                ts_static_only=args.ts_static_only,
                force=args.force,
                dry_run=args.dry_run,
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user. Stopping after current triple.")
            break
        except Exception as e:
            res = TripleResult(cs=cs, error=f"Unexpected exception: {e}")
            res.aptrees_status = "failed"
        results.append(res)

        # Running summary so progress is visible during long runs.
        n_done = sum(1 for r in results if r.merge_status == "ok"
                     or "skipped" in r.merge_status)
        n_failed = sum(1 for r in results if "failed" in (r.merge_status or "")
                       or "failed" in (r.aptrees_status or "")
                       or "failed" in (r.triplesort_status or ""))
        wall = (time.monotonic() - overall_start) / 60
        print(f"  PROGRESS: {i}/{len(triples)} processed, {n_done} done, "
              f"{n_failed} failed, {wall:.1f} min elapsed total")

    # Final summary table.
    print(f"\n{'='*72}")
    print("FINAL SUMMARY")
    print(f"{'='*72}")
    print(f"{'ID':>3} {'TRIPLE':<28} {'AP':<8} {'TS':<8} {'MERGE':<8} {'MIN':>6}")
    for r in results:
        print(f"{r.cs.id:>3} {_triple_dir_name(r.cs):<28} "
              f"{r.aptrees_status:<8} {r.triplesort_status:<8} "
              f"{r.merge_status:<8} {r.elapsed_sec/60:>6.1f}")
        if r.error:
            print(f"     error: {r.error}")
    print(f"\nTotal wall time: {(time.monotonic() - overall_start)/60:.1f} min")


if __name__ == "__main__":
    main()