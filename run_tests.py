"""
run_tests.py

Runs Hestia's full test suite (everything under tests/) — a drop-in
stand-in for `python -m pytest` that also guarantees the repo root is on
sys.path, so `import modules.x` / `import core.x` resolve correctly no
matter what directory you happen to invoke this from.

Usage:
    python run_tests.py                    # run everything
    python run_tests.py -v                 # verbose
    python run_tests.py -x                 # stop at first failure
    python run_tests.py tests/test_hermes.py         # just one file
    python run_tests.py -k "confirm or cancel"       # pytest -k expression
    python run_tests.py --lf                # only rerun last failures

Any arguments you pass are forwarded straight to pytest, so every normal
pytest flag/expression works exactly as it would with `pytest ...` — this
just removes the need to remember to `cd` into the repo root or use
`python -m` for imports to resolve.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_TESTS_DIR = _REPO_ROOT / "tests"


def main() -> int:
    # `python -m pytest` gets the repo root on sys.path for free (that's
    # what `-m` does); running this file directly needs it done by hand,
    # or `modules.pluto...`/`core.stt` imports inside the test files would
    # fail depending on CWD.
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    try:
        import pytest
    except ImportError:
        print(
            "pytest isn't installed in this environment.\n"
            "Install the test dependencies first, e.g.:\n"
            "    pip install pytest\n"
            "(or whatever subset of requirements.txt your test run needs).",
            file=sys.stderr,
        )
        return 1

    if not _TESTS_DIR.is_dir():
        print(f"No tests/ directory found at {_TESTS_DIR}", file=sys.stderr)
        return 1

    # Any CLI args the user passed (file paths, -k, -v, -x, --lf, ...) take
    # over entirely; with none, default to running the whole tests/ dir.
    args = sys.argv[1:] or [str(_TESTS_DIR)]
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())