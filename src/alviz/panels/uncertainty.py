"""Ensemble uncertainty across active-learning iterations.

* ``UncertaintyData``           -- the data contract.
* ``SampleUncertainties``       -- abstract: heatmaps + contract.
* ``ExtxyzSampleUncertainties`` -- reads per-iteration uncertainties from
                                   extxyz files.

Renders two heatmaps -- energy and force uncertainty -- over (iteration
index, configuration index). Iterations may hold different numbers of
configurations; shorter rows are NaN-padded (rendered blank).
"""

from abc import abstractmethod

import numpy as np

from alviz.dashboard import Entry
from alviz.data import PanelData


def _pad_to_grid(rows):
    """Stack ragged rows into a 2D array, padding short rows with NaN."""
    if not rows:
        return np.zeros((0, 0))
    width = max(len(r) for r in rows)
    grid = np.full((len(rows), width), np.nan)
    for i, row in enumerate(rows):
        grid[i, : len(row)] = row
    return grid


class UncertaintyData(PanelData):
    """Per-iteration lists of energy/force uncertainty (possibly ragged)."""

    euncs: list[list[float]]
    funcs: list[list[float]]
    n_iterations: int


class SampleUncertainties(Entry):
    """Energy/force ensemble-uncertainty heatmaps per AL iteration.

    Args (rendering):
        orientation: ``"horizontal"`` (side by side) or ``"vertical"``.
        cmap: Matplotlib colormap.
        vmax: Upper color limit (shared); None = autoscale.
    """

    name = "sample_uncertainties"
    data_class = UncertaintyData

    def __init__(
        self,
        *args,
        orientation: str = "horizontal",
        cmap: str = "viridis",
        vmax=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._cmap = cmap
        self._vmax = vmax
        if orientation == "vertical":
            self._grid = (2, 1)
        elif orientation == "horizontal":
            self._grid = (1, 2)
        else:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(*self._grid)
        self.ax_e = fig.add_subplot(self.gs[0])
        self.ax_e.set_title("Energy uncertainty")
        self.ax_f = fig.add_subplot(self.gs[1])
        self.ax_f.set_title("Force uncertainty")

    @abstractmethod
    def compute_data(self, shared_ressources: dict) -> UncertaintyData:
        """Produce per-iteration uncertainties (see :class:`UncertaintyData`)."""
        raise NotImplementedError

    def _heatmap(self, ax, rows, label):
        grid = _pad_to_grid(rows)
        if grid.size == 0:
            return
        im = ax.pcolormesh(
            grid, cmap=self._cmap, vmin=0, vmax=self._vmax, rasterized=True
        )
        self._fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("configuration index")
        ax.set_ylabel("iteration")

    def plot_data(self):
        self._heatmap(self.ax_e, self.data.euncs, "energy uncert.")
        self._heatmap(self.ax_f, self.data.funcs, "force uncert.")


class ExtxyzSampleUncertainties(SampleUncertainties):
    """Read per-iteration uncertainties from one extxyz file per iteration.

    Each configuration carries the uncertainties in ``atoms.info`` under
    ``energy_uncert_key`` / ``forces_uncert_key`` (both overridable).

    Args:
        traj_files: Ordered list of per-iteration extxyz files.
    """

    def __init__(
        self,
        traj_files,
        *args,
        energy_uncert_key: str = "energy_uncert",
        forces_uncert_key: str = "forces_uncert_normed",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._traj_files = list(traj_files)
        self._energy_uncert_key = energy_uncert_key
        self._forces_uncert_key = forces_uncert_key

    def compute_data(self, shared_ressources: dict) -> UncertaintyData:
        import ase.io

        euncs, funcs = [], []
        for path in self._traj_files:
            configs = ase.io.read(path, index=":")
            euncs.append(
                [float(at.info[self._energy_uncert_key]) for at in configs]
            )
            funcs.append(
                [float(at.info[self._forces_uncert_key]) for at in configs]
            )

        return UncertaintyData(
            euncs=euncs,
            funcs=funcs,
            n_iterations=len(self._traj_files),
        )
