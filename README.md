# alviz

Grid-dashboard visualization for **active-learning machine-learned
force-field (MLFF) workflows**.

`alviz` is a standalone, *downstream* visualization package. It depends on
the packages that produce data (e.g. `remote_handler`, `jaxnest`) — never the
other way around — and panels couple to **producer artifacts** (files with a
documented schema), not to producer internals. This keeps responsibilities
cleanly separated:

- **orchestration** (`remote_handler`) and **method/science** (`jaxnest`)
  *produce* artifacts;
- **alviz** *consumes* them and renders dashboards.

## Architecture

- `alviz.dashboard.Entry` — abstract panel: `allocate_panels(fig)` →
  `compute_data(shared) -> JsonDict` → `plot_data()`, with per-panel JSON
  caching so expensive computations are not repeated.
- `alviz.dashboard.Dashboard` — arranges entries on a matplotlib grid and
  drives the compute → plot → save lifecycle.
- `alviz.mpl_utils.JsonDict` — JSON-serializable mapping; the on-disk
  contract for panel data.

Concrete panels (reading `AL_progress.json`, `*.extxyz`, model manifests,
…) live in `alviz.panels` and are added incrementally; each declares its own
optional dependencies (e.g. `ase`, `scikit-learn`) via extras.

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from alviz import Dashboard, Entry, JsonDict


class MyPanel(Entry):
    def allocate_panels(self, fig):
        self.ax = fig.add_subplot(self.subspec)

    def compute_data(self, shared):
        return JsonDict({"x": [0, 1, 2], "y": [0, 1, 4]})

    def plot_data(self):
        self.ax.plot(self.data["x"], self.data["y"])


board = Dashboard(1, 1)
board.register_entry(MyPanel(datapath="."), ("0", "0"))
board.allocate_panels()
board.compute_data()
board.plot_panels()
board.savefig("board.pdf")
```

## Test

```bash
pytest
```
