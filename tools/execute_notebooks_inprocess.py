"""Execute self-contained analysis notebooks when local Jupyter sockets are unavailable."""

from __future__ import annotations

import contextlib
import base64
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat


def execute(path: Path) -> None:
    book = nbformat.read(path, as_version=4)
    scope = {"__name__": "__main__"}
    for number, cell in enumerate(book.cells, start=1):
        if cell.cell_type != "code":
            continue
        outputs = []

        def display(*values):
            for value in values:
                if hasattr(value, "_repr_html_"):
                    data = {"text/plain": str(value), "text/html": value._repr_html_()}
                else:
                    data = {"text/plain": repr(value)}
                outputs.append(nbformat.v4.new_output("display_data", data=data))

        def show(*args, **kwargs):
            for figure_number in plt.get_fignums():
                figure = plt.figure(figure_number)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
                outputs.append(nbformat.v4.new_output(
                    "display_data", data={"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")}))
                plt.close(figure)

        scope["display"] = display
        plt.show = show
        stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(stream):
                exec(compile(cell.source, f"{path.name}:cell{number}", "exec"), scope)
        except Exception as exc:
            print(f"FAILED: {path.name}, code cell {number}: {exc}", file=sys.stderr)
            raise
        output = stream.getvalue()
        if output:
            # Keep the final messages; progress bars remain active on rerun.
            outputs.insert(0, nbformat.v4.new_output("stream", name="stdout", text=output[-5000:]))
        cell.outputs = outputs
        cell.execution_count = number
        print(f"Executed {path.name} code cell {number}", flush=True)
    nbformat.validate(book)
    nbformat.write(book, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        execute(Path(argument))
