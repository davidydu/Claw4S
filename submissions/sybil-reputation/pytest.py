"""Minimal pytest-compatible test runner for offline execution.

Supports this submission's existing function-style tests and basic CLI flags:
``python -m pytest``, ``python -m pytest tests/``, and ``-v``.
"""

from __future__ import annotations

import argparse
import runpy
import traceback
from pathlib import Path
from typing import Callable, Iterable, List, Tuple


TestCase = Tuple[str, Callable[[], None]]


def _iter_test_files(paths: List[str]) -> Iterable[Path]:
    if not paths:
        paths = ["tests"]

    files: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.name.startswith("test_") and path.suffix == ".py":
            files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(path.rglob("test_*.py")))
    return sorted(files)


def _discover_tests(files: Iterable[Path]) -> List[TestCase]:
    tests: List[TestCase] = []
    for file_path in files:
        namespace = runpy.run_path(str(file_path))
        for name, obj in sorted(namespace.items()):
            if name.startswith("test_") and callable(obj):
                test_id = f"{file_path.as_posix()}::{name}"
                tests.append((test_id, obj))
    return tests


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args, _unknown = parser.parse_known_args()

    tests = _discover_tests(_iter_test_files(args.paths))
    passed = 0
    failed = 0

    for test_id, test_fn in tests:
        try:
            test_fn()
            passed += 1
            if args.verbose and not args.quiet:
                print(f"PASSED {test_id}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAILED {test_id}")
            traceback.print_exc()

    if not args.quiet:
        if failed == 0:
            print(f"{passed} passed")
        else:
            print(f"{failed} failed, {passed} passed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
