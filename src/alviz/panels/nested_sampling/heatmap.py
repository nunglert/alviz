"""Thermodynamic heatmap over a temperature/pressure grid.

* ``HeatmapData``  -- the data contract (T/P axes + a 2D observable grid).
* ``HeatmapPanel`` -- abstract: the heatmap rendering + contract.

There is no ``Extxyz*`` reader: the grid is produced by postprocessing (e.g.
nested-sampling observables), which is the project's responsibility. A
project subclasses ``HeatmapPanel`` and implements ``compute_data``.
"""

from abc import abstractmethod
from typing import Optional

import numpy as np
from pydantic import model_validator

from alviz.dashboard import Entry
from alviz.data import PanelData


class HeatmapData(PanelData):
    """A 2D observable over a (pressure, temperature) grid.

    ``obs`` has shape ``(len(p), len(t))`` -- one row per pressure.
    """

    t: list[float]
    p: list[float]
    obs: list[list[float]]
    quantity: Optional[str] = None

    @model_validator(mode="after")
    def _check_shape(self):
        if len(self.obs) != len(self.p):
            raise ValueError("obs must have one row per pressure (len(p))")
        if self.obs and len(self.obs[0]) != len(self.t):
            raise ValueError(
                "obs rows must have one column per temperature (len(t))"
            )
        return self


def heatmap_pcolormesh(
    ax,
    t,
    p,
    obs,
    *,
    cmap: str = "viridis",
    cut_min=None,
    cut_max=None,
    temp_scale: float = 1.0,
    obs_scale: float = 1.0,
    T_label: str = r"$T$",
    P_label: str = r"$\log P$",
    label=None,
    do_colorbar: bool = True,
):
    """Render a (pressure, temperature) observable grid as a pcolormesh."""
    t = np.asarray(t, dtype=float) * temp_scale
    p = np.asarray(p, dtype=float)
    obs = np.asarray(obs, dtype=float) * obs_scale
    im = ax.pcolormesh(
        t,
        p,
        obs,
        cmap=cmap,
        vmin=cut_min,
        vmax=cut_max,
        shading="auto",
        rasterized=True,
    )
    ax.set_xlabel(T_label)
    ax.set_ylabel(P_label)
    if do_colorbar:
        ax.figure.colorbar(im, ax=ax, label=label)
    return im


class HeatmapPanel(Entry):
    """A temperature/pressure heatmap of a thermodynamic observable.

    Args (rendering):
        plot_kwargs: forwarded to :func:`heatmap_pcolormesh` (``cmap``,
            ``cut_min``/``cut_max``, ``T_label``/``P_label``, ``label``, ...).
    """

    name = "heatmap"
    data_class = HeatmapData

    def __init__(self, *args, plot_kwargs: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._plot_kwargs = plot_kwargs or {}

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    @abstractmethod
    def compute_data(self, shared_ressources: dict) -> HeatmapData:
        """Produce the T/P observable grid (see :class:`HeatmapData`)."""
        raise NotImplementedError

    def plot_data(self):
        d = self.data
        kw = dict(self._plot_kwargs)
        if d.quantity and "label" not in kw:
            kw["label"] = d.quantity
        heatmap_pcolormesh(self.ax, d.t, d.p, d.obs, **kw)
        if d.quantity:
            self.ax.set_title(d.quantity)
