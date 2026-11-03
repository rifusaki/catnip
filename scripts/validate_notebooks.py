"""Compile-check every code cell of a Jupyter notebook.

Strips IPython shell/magic lines (leading ``!`` or ``%``) and validates the
remaining source of each code cell with ``ast.parse`` + ``compile``.
Exits non-zero if any cell fails to compile.

Usage:
    python scripts/validate_notebooks.py [notebook.ipynb ...]

With no arguments, validates ``notebooks/catnipColab.ipynb``.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK = REPO_ROOT / "notebooks" / "catnipColab.ipynb"


def strip_ipython_lines(source: str) -> str:
    """Drop lines that are IPython shell (``!``) or magic (``%``) commands."""
    kept = [
        line
        for line in source.splitlines()
        if not line.lstrip().startswith(("!", "%"))
    ]
    return "\n".join(kept)


def validate_notebook(path: Path) -> tuple[int, int, list[str]]:
    """Return (ok_count, total_code_cells, error_messages) for one notebook."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    total = 0
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        total += 1
        cell_id = cell.get("id", f"cell-{index}")
        source = strip_ipython_lines("".join(cell.get("source", [])))
        filename = f"{path}:cell-{index}({cell_id})"
        try:
            ast.parse(source, filename=filename)
            compile(source, filename, "exec")
        except SyntaxError as exc:
            errors.append(f"{filename} line {exc.lineno}: {exc.msg}")
    return total - len(errors), total, errors


def main(argv: list[str]) -> int:
    paths = [Path(arg) for arg in argv] or [DEFAULT_NOTEBOOK]
    for path in paths:
        if not path.is_file():
            print(f"ERROR: notebook not found: {path}", file=sys.stderr)
            return 1

    total_ok = 0
    total_cells = 0
    all_errors: list[str] = []
    for path in paths:
        ok, total, errors = validate_notebook(path)
        total_ok += ok
        total_cells += total
        all_errors.extend(errors)
        status = "OK" if not errors else "FAIL"
        print(f"{path}: {ok}/{total} code cells compile [{status}]")
        for err in errors:
            print(f"  {err}")

    print(f"TOTAL: {total_ok}/{total_cells} cells compiled, {len(all_errors)} errors")
    if all_errors:
        return 1
    print("100% of code cells compile cleanly, 0 errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
