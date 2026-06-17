"""Combine several single-page PDFs into one multipage PDF.

A :class:`~alviz.dashboard.Dashboard` renders to a *single* matplotlib figure,
i.e. one page. To produce a multipage report -- e.g. one dashboard page per
active-learning iteration -- save each page on its own and stitch them together
here. Kept in alviz (not the consumers) so the PDF-merge dependency lives with
the dashboard engine.
"""

from pathlib import Path

from pypdf import PdfWriter


def merge_pdfs(inputs, output):
    """Concatenate ``inputs`` (in order) into one PDF at ``output``.

    Args:
        inputs: Ordered iterable of PDF paths; each becomes one or more pages
            in the result, in the given order.
        output: Destination PDF path. Parent directories are created.

    Returns:
        The ``output`` path as a :class:`pathlib.Path`.

    Raises:
        ValueError: If ``inputs`` is empty (nothing to merge).
    """
    inputs = [Path(p) for p in inputs]
    if not inputs:
        raise ValueError("merge_pdfs: no input PDFs given.")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = PdfWriter()
    try:
        for path in inputs:
            writer.append(str(path))
        with open(output, "wb") as fh:
            writer.write(fh)
    finally:
        writer.close()
    return output
