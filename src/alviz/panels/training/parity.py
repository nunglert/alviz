"""Energy/force parity (predicted vs. reference).

Structure (the pattern for all "basic" panels):

* ``ParityData``       -- the typed data contract (what gets plotted).
* ``ParityPanel``      -- abstract: owns the *rendering* + the contract;
                          ``compute_data`` is left to subclasses.
* ``ExtxyzParityPanel``-- batteries-included: produces ``ParityData`` by
                          reading extxyz files (the common case).

alviz owns the plot and the contract; *where the data comes from* is the
subclass's business. A project that needs predictions from a model, a
database, or anything else subclasses ``ParityPanel`` and implements
``compute_data`` -- and gets the plot + validation for free.
"""

from abc import abstractmethod
from typing import Optional

import numpy as np
from pydantic import model_validator

from alviz.dashboard import Entry
from alviz.data import PanelData
from alviz.panels._atoms import as_paths, resolve_array, resolve_scalar


# --------------------------------------------------------------------------
# Data contract
# --------------------------------------------------------------------------
class ParityData(PanelData):
    """Reference vs. predicted energies and (optionally) forces.

    Energies are per-config scalars; forces are flattened component lists.
    """

    e_obs: list[float]
    e_pred: list[float]
    e_rmse: float
    e_mae: float
    per_atom: bool
    f_obs: Optional[list[float]] = None
    f_pred: Optional[list[float]] = None
    f_rmse: Optional[float] = None
    f_mae: Optional[float] = None

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.e_obs) != len(self.e_pred):
            raise ValueError("e_obs and e_pred must have equal length")
        if self.f_obs is not None and self.f_pred is not None:
            if len(self.f_obs) != len(self.f_pred):
                raise ValueError("f_obs and f_pred must have equal length")
        return self


# --------------------------------------------------------------------------
# Plot helper + stats
# --------------------------------------------------------------------------
def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a, b):
    return float(np.mean(np.abs(a - b)))


def parity_scatter(
    ax,
    obs,
    pred,
    *,
    rmse=None,
    mae=None,
    xlabel="",
    ylabel="",
    title=None,
    **scatter_kwargs,
):
    """Scatter ``pred`` vs ``obs`` with a y=x guide and RMSE/MAE annotation."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    s = scatter_kwargs.pop("s", 8)
    alpha = scatter_kwargs.pop("alpha", 0.5)
    ax.scatter(obs, pred, s=s, alpha=alpha, **scatter_kwargs)

    if obs.size and pred.size:
        lo = float(min(obs.min(), pred.min()))
        hi = float(max(obs.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], color="k", lw=1, ls="--", zorder=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    lines = []
    if rmse is not None:
        lines.append(f"RMSE = {rmse:.4g}")
    if mae is not None:
        lines.append(f"MAE = {mae:.4g}")
    if lines:
        ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )
    ax.set_aspect("equal", adjustable="datalim")


# --------------------------------------------------------------------------
# Abstract panel: rendering + contract; compute_data is downstream
# --------------------------------------------------------------------------
class ParityPanel(Entry):
    """Predicted-vs-reference parity. Subclasses produce :class:`ParityData`.

    Args (rendering):
        show_forces: Include the forces parity subplot.
        plot_kwargs: Per-axis scatter kwargs ``{"energy": {...},
            "forces": {...}}``.
    """

    name = "parity"
    data_class = ParityData
    colspan = 2

    def __init__(self, *args, show_forces=True, plot_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._show_forces = show_forces
        self._plot_kwargs = plot_kwargs or {}

    def allocate_panels(self, fig):
        self._fig = fig
        if self._show_forces:
            self.gs = self.subspec.subgridspec(1, 2)
            self.ax_e = fig.add_subplot(self.gs[0])
            self.ax_f = fig.add_subplot(self.gs[1])
        else:
            self.ax_e = fig.add_subplot(self.subspec)
            self.ax_f = None

    @abstractmethod
    def compute_data(self, shared_ressources: dict) -> ParityData:
        """Produce the parity data (see :class:`ParityData`)."""
        raise NotImplementedError

    def plot_data(self):
        d = self.data
        e_unit = "eV/atom" if d.per_atom else "eV"
        parity_scatter(
            self.ax_e,
            d.e_obs,
            d.e_pred,
            rmse=d.e_rmse,
            mae=d.e_mae,
            xlabel=f"reference E ({e_unit})",
            ylabel=f"predicted E ({e_unit})",
            title=self.name,
            **self._plot_kwargs.get("energy", {}),
        )
        if self.ax_f is not None and d.f_obs is not None:
            parity_scatter(
                self.ax_f,
                d.f_obs,
                d.f_pred,
                rmse=d.f_rmse,
                mae=d.f_mae,
                xlabel="reference F (eV/Å)",
                ylabel="predicted F (eV/Å)",
                **self._plot_kwargs.get("forces", {}),
            )


# --------------------------------------------------------------------------
# Concrete reader: produce ParityData from extxyz (the common case)
# --------------------------------------------------------------------------
class ExtxyzParityPanel(ParityPanel):
    """Read reference + predicted labels from extxyz files.

    Each configuration must carry both a reference and a predicted value;
    keys default to ``REF_energy``/``REF_forces`` (reference) and
    ``energy``/``forces`` (prediction, resolved from info/arrays *or* the
    attached calculator), all overridable.
    """

    def __init__(
        self,
        config_files,
        *args,
        energy_ref_key: str = "REF_energy",
        energy_pred_key: str = "energy",
        forces_ref_key: str = "REF_forces",
        forces_pred_key: str = "forces",
        per_atom: bool = True,
        show_forces: bool = True,
        plot_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            show_forces=show_forces,
            plot_kwargs=plot_kwargs,
            **kwargs,
        )
        self._config_files = as_paths(config_files)
        self._energy_ref_key = energy_ref_key
        self._energy_pred_key = energy_pred_key
        self._forces_ref_key = forces_ref_key
        self._forces_pred_key = forces_pred_key
        self._per_atom = per_atom

    def compute_data(self, shared_ressources: dict) -> ParityData:
        import ase.io

        e_obs, e_pred = [], []
        f_obs, f_pred = [], []

        for path in self._config_files:
            for atoms in ase.io.read(path, index=":"):
                norm = len(atoms) if self._per_atom else 1
                e_obs.append(
                    resolve_scalar(atoms, self._energy_ref_key) / norm
                )
                e_pred.append(
                    resolve_scalar(atoms, self._energy_pred_key) / norm
                )
                if self._show_forces:
                    f_obs.extend(
                        resolve_array(atoms, self._forces_ref_key).ravel()
                    )
                    f_pred.extend(
                        resolve_array(atoms, self._forces_pred_key).ravel()
                    )

        e_obs = np.asarray(e_obs, dtype=float)
        e_pred = np.asarray(e_pred, dtype=float)

        fields = dict(
            e_obs=e_obs.tolist(),
            e_pred=e_pred.tolist(),
            e_rmse=_rmse(e_obs, e_pred),
            e_mae=_mae(e_obs, e_pred),
            per_atom=self._per_atom,
        )
        if self._show_forces:
            f_obs = np.asarray(f_obs, dtype=float)
            f_pred = np.asarray(f_pred, dtype=float)
            fields.update(
                f_obs=f_obs.tolist(),
                f_pred=f_pred.tolist(),
                f_rmse=_rmse(f_obs, f_pred),
                f_mae=_mae(f_obs, f_pred),
            )
        return ParityData(**fields)
