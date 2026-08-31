"""Execute the notebook from a fresh kernel and export a readable HTML copy."""
from pathlib import Path
import os
import argparse
import tempfile
import io
import contextlib
import traceback
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import AsyncKernelManager
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]


def execute_in_process(notebook):
    """Execute every cell in a fresh process using IPython, without socket IPC.

    This captures real stdout and rich display outputs. It is useful in runtime
    environments that prohibit kernel sockets; it does not synthesize results.
    """
    shell = InteractiveShell.instance()
    shell.user_ns.update({"__name__": "__main__"})
    os.chdir(ROOT)
    counter = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        counter += 1
        cell.execution_count = counter
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        error = result.error_before_exec or result.error_in_exec
        if error:
            raise RuntimeError(f"Notebook cell {counter} failed: {error}") from error
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for item in captured.outputs:
            outputs.append(nbformat.v4.new_output("display_data", data=item.data, metadata=item.metadata))
        cell.outputs = outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipc", action="store_true", help="Use local IPC transport when TCP sockets are unavailable")
    parser.add_argument("--in-process", action="store_true", help="Execute with IPython without requiring kernel sockets")
    args = parser.parse_args()
    path = ROOT / "notebooks/customer_segmentation.ipynb"
    notebook = nbformat.read(path, as_version=4)
    if args.in_process:
        execute_in_process(notebook)
    else:
        manager = AsyncKernelManager(kernel_name="python3", transport="ipc", ip=str(Path(tempfile.gettempdir()) / f"customer-segmentation-{os.getpid()}")) if args.ipc else None
        client = NotebookClient(notebook, km=manager, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
        client.execute(cleanup_kc=True)
    nbformat.write(notebook, path)
    assert not any(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []))
    body, _ = HTMLExporter().from_notebook_node(notebook)
    (ROOT / "notebooks/customer_segmentation.html").write_text(body, encoding="utf-8")
    count = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Executed {count} code cells without errors and exported HTML.")
