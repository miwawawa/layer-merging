"""Lightweight notebook runner to verify every notebook executes end-to-end.

Executes each code cell in order (no magics used in the notebooks), capturing
any exception. Matplotlib is forced to the Agg backend.
"""

import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")

import nbformat


def run(name: str, cwd: str) -> int:
    path = os.path.join(cwd, name)
    nb = nbformat.read(path, as_version=4)
    failures = 0
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source
        if src.startswith("%"):
            continue
        try:
            exec(compile(src, f"{name}#cell{i}", "exec"), ns)   # shared namespace
        except Exception:
            print(f"FAIL {name} cell {i}:\n{traceback.format_exc()}")
            failures += 1
    return failures


def main():
    here = sys.argv[1] if len(sys.argv) > 1 else "notebooks"
    total = 0
    cwd_save = os.getcwd()
    for name in sorted(os.listdir(here)):
        if not name.endswith(".ipynb"):
            continue
        sys.path.insert(0, os.path.abspath(os.path.join(here, "..", "src")))
        os.chdir(here)
        full = os.path.join(os.getcwd(), name)
        fails = run(full, os.getcwd())
        os.chdir(cwd_save)
        print(f"{name}: {'OK' if fails == 0 else f'{fails} CELLS FAILED'}")
        total += fails
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()