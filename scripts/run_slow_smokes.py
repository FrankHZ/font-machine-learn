from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "tests" / "test_nftr_export.py"
TEST_CLASS = "tests.test_nftr_export.NFTRExportTest"


@dataclass(frozen=True)
class TestRun:
    test_id: str
    returncode: int
    seconds: float
    output: str


def slow_test_ids(test_file: Path = TEST_FILE) -> list[str]:
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    ids: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "NFTRExportTest":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or not item.name.startswith("test_"):
                continue
            decorator_names = {
                decorator.id
                for decorator in item.decorator_list
                if isinstance(decorator, ast.Name)
            }
            if "slow_test" in decorator_names:
                ids.append(f"{TEST_CLASS}.{item.name}")
    return ids


def matches_patterns(test_id: str, patterns: list[str]) -> bool:
    return not patterns or any(pattern in test_id for pattern in patterns)


def run_one(test_id: str) -> TestRun:
    env = os.environ.copy()
    env["FML_RUN_SLOW_TESTS"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", "-m", "unittest", test_id]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return TestRun(
        test_id=test_id,
        returncode=completed.returncode,
        seconds=time.perf_counter() - started,
        output=completed.stdout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run @slow_test unittest methods in parallel worker processes.")
    parser.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--pattern", action="append", default=[], help="Only run test ids containing this substring.")
    parser.add_argument("--list", action="store_true", help="List selected slow tests without running them.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop scheduling output after the first failure is seen.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = [test_id for test_id in slow_test_ids() if matches_patterns(test_id, args.pattern)]
    if args.list:
        for test_id in selected:
            print(test_id)
        print(f"selected={len(selected)}")
        return 0
    if not selected:
        print("No slow tests matched.", file=sys.stderr)
        return 2

    started = time.perf_counter()
    failures: list[TestRun] = []
    completed_runs: list[TestRun] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {executor.submit(run_one, test_id): test_id for test_id in selected}
        for future in as_completed(futures):
            result = future.result()
            completed_runs.append(result)
            status = "ok" if result.returncode == 0 else "FAIL"
            print(f"[{status}] {result.seconds:7.2f}s {result.test_id}")
            if result.returncode != 0:
                failures.append(result)
                print(result.output)
                if args.fail_fast:
                    break

    elapsed = time.perf_counter() - started
    print(f"completed={len(completed_runs)} selected={len(selected)} jobs={max(1, args.jobs)} seconds={elapsed:.2f}")
    if failures:
        print(f"failures={len(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
