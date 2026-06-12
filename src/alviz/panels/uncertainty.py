"""Ensemble uncertainty across active-learning iterations.

Pure reader: given one extxyz file per AL iteration (each holding the
candidate configurations of that iteration, labelled with per-config
ensemble uncertainties), it renders two heatmaps -- energy and force
uncertainty -- over (iteration index, configuration index).

Artifact contract
-----------------
Each configuration carries, in ``atoms.info``:
    * ``energy_uncert``          -- scalar energy uncertainty
    * ``forces_uncert_normed``   -- scalar (norm-reduced) force uncertainty

Both key names are overridable. Iterations may contain different numbers of
configurations; shorter rows are padded with NaN (rendered blank).
"""

import numpy as np

from alviz.dashboard import Entry
from alviz.mpl_utils import JsonDict


def _pad_to_grid(rows):
    """Stack ragged rows into a 2D array, padding short rows with NaN."""
    if not rows:
        return np.zeros((0, 0))
    width = max(len(r) for r in rows)
    grid = np.full((len(rows), width), np.nan)
    for i, row in enumerate(rows):
        grid[i, : len(row)] = row
    return grid


class SampleUncertainties(Entry):
    """Energy/force ensemble-uncertainty heatmaps per AL iteration.

    Args:
        traj_files: Ordered list of per-iteration extxyz files.
        energy_uncert_key: ``info`` key for the energy uncertainty.
        forces_uncert_key: ``info`` key for the force uncertainty.
        orientation: ``"horizontal"`` (side by side) or ``"vertical"``.
        cmap: Matplotlib colormap.
        vmax: Upper color limit (shared by both panels); None = autoscale.
    """

    name = "sample_uncertainties"

    def __init__(
        self,
        traj_files,
        *args,
        energy_uncert_key: str = "energy_uncert",
        forces_uncert_key: str = "forces_uncert_normed",
        orientation: str = "horizontal",
        cmap: str = "viridis",
        vmax=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._traj_files = list(traj_files)
        self._energy_uncert_key = energy_uncert_key
        self._forces_uncert_key = forces_uncert_key
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

    def compute_data(self, shared_ressources: dict) -> JsonDict:
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

        return JsonDict(
            euncs=euncs,
            funcs=funcs,
            n_iterations=len(self._traj_files),
        )

    def _heatmap(self, ax, rows, label):
        grid = _pad_to_grid(rows)
        if grid.size == 0:
            return
        im = ax.pcolormesh(
            grid,
            cmap=self._cmap,
            vmin=0,
            vmax=self._vmax,
            rasterized=True,
        )
        self._fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("configuration index")
        ax.set_ylabel("iteration")

    def plot_data(self):
        self._heatmap(self.ax_e, self.data["euncs"], "energy uncert.")
        self._heatmap(self.ax_f, self.data["funcs"], "force uncert.")
