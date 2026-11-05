"""Execute notebooks/catnipBenchmarkPresentation.ipynb headlessly.

Reads the notebook, executes all cells against local or R2-downloaded benchmark
artifacts using the exact Python interpreter of the active environment, and writes
the executed output (tables, charts, visual panels) back to the .ipynb file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nbclient
import nbformat
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
from jupyter_client.manager import AsyncKernelManager

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_NOTEBOOK = REPO_ROOT / "notebooks" / "catnipBenchmarkPresentation.ipynb"


class CurrentEnvKernelSpecManager(KernelSpecManager):
    """Enforces execution with sys.executable, bypassing machine-local kernels."""

    def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
        return KernelSpec(
            argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            display_name="Current Environment Python",
            language="python",
        )

    def find_kernel_specs(self) -> dict[str, KernelSpec]:
        return {"python3": self.get_kernel_spec("python3")}


def execute_notebook(
    notebook_path: Path,
    output_path: Path | None = None,
    timeout: int = 600,
) -> None:
    """Execute all cells in the notebook and save back to disk."""
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    destination = output_path or notebook_path
    print(f"[ExecNotebook] Reading {notebook_path}...")
    nb = nbformat.read(notebook_path, as_version=4)

    print(f"[ExecNotebook] Using interpreter: {sys.executable}")
    print(f"[ExecNotebook] Executing cells (timeout={timeout}s, cwd={REPO_ROOT})...")
    km = AsyncKernelManager(kernel_spec_manager=CurrentEnvKernelSpecManager())
    client = nbclient.NotebookClient(
        nb,
        km=km,
        timeout=timeout,
        record_timing=True,
    )
    client.execute(cwd=str(REPO_ROOT))

    print(f"[ExecNotebook] Writing executed notebook to {destination}...")
    nbformat.write(nb, destination)
    print("[ExecNotebook] Notebook execution finished cleanly.")


def verify_outputs(notebook_path: Path) -> None:
    """Verify that all code cells ran and none produced error outputs."""
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"\n[Verification] Validating {len(code_cells)} code cell outputs...")

    for i, cell in enumerate(code_cells):
        outputs = cell.get("outputs", [])
        output_types = [o.get("output_type") for o in outputs]
        print(f"  Code Cell {i+1}: {len(outputs)} output(s) -> {output_types}")

        errors = [o for o in outputs if o.get("output_type") == "error"]
        if errors:
            err = errors[0]
            print(
                f"    ERROR in Cell {i+1}: {err.get('ename')}: {err.get('evalue')}",
                file=sys.stderr,
            )
            for tb_line in err.get("traceback", []):
                print(f"      {tb_line}", file=sys.stderr)
            raise RuntimeError(f"Cell {i+1} failed with execution error!")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Headlessly execute the Catnip benchmark presentation notebook."
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=DEFAULT_NOTEBOOK,
        help="Path to source notebook (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output notebook (default: overwrite source notebook)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-cell timeout in seconds (default: %(default)s)",
    )

    args = parser.parse_args(argv)
    output_target = args.output or args.notebook

    try:
        execute_notebook(args.notebook, output_path=output_target, timeout=args.timeout)
        verify_outputs(output_target)
        return 0
    except Exception as exc:
        print(f"[ExecNotebook] FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
