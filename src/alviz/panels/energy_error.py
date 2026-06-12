"""Per-config energy error against a reference, vs. reference energy.

* ``EnergyErrorData``        -- the data contract.
* ``EnergyErrorPanel``       -- abstract: the error plot + contract.
* ``ExtxyzEnergyErrorPanel`` -- reads ref + predicted energy from extxyz.

A benchmark view: for each configuration, the absolute predicted-minus-
reference error (red, left axis) and the predicted energy (blue, right axis)
against the reference energy. This is the pure-reader form of the original
``BenchmarkDatabase`` panel (the model-evaluation branch is dropped --
producing predictions is a producer/project responsibility).
"""

from abc import abstractmethod

import numpy as np
from pydantic import model_validator

from alviz.dashboard import Entry
from alviz.data import PanelData
from alviz.panels._atoms import as_paths, resolve_scalar


class EnergyErrorData(PanelData):
    """Reference and predicted per-config energies."""

    es_ref: list[float]
    es_pred: list[float]
    per_atom: bool

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.es_ref) != len(self.es_pred):
            raise ValueError("es_ref and es_pred must have equal length")
        return self


class EnergyErrorPanel(Entry):
    """Energy error and predicted energy vs. reference energy."""

    name = "energy_error"
    data_class = EnergyErrorData

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])
        self.ax.set_title(self.name)

    @abstractmethod
    def compute_data(self, shared_ressources: dict) -> EnergyErrorData:
        """Produce reference/predicted energies (see :class:`EnergyErrorData`)."""
        raise NotImplementedError

    def plot_data(self):
        d = self.data
        unit = "eV atom$^{-1}$" if d.per_atom else "eV"
        es_ref = np.asarray(d.es_ref, dtype=float)
        es_pred = np.asarray(d.es_pred, dtype=float)
        err = np.abs(es_pred - es_ref)

        ecolor = "tab:blue"
        errcolor = "tab:red"

        self.ax.set_xlabel(rf"$E_\mathrm{{ref}}$ [{unit}]")
        self.ax.set_ylabel(rf"$\Delta$ [{unit}]", color=errcolor)
        self.ax.scatter(es_ref, err, s=5, color=errcolor)
        self.ax.tick_params(axis="y", labelcolor=errcolor)

        ax2 = self.ax.twinx()
        ax2.set_ylabel(rf"$E_\mathrm{{pred}}$ [{unit}]", color=ecolor)
        ax2.scatter(es_ref, es_pred, s=5, color=ecolor)
        ax2.tick_params(axis="y", labelcolor=ecolor)


class ExtxyzEnergyErrorPanel(EnergyErrorPanel):
    """Read reference + predicted per-config energies from extxyz files.

    Args:
        config_files: One path or list of paths to labelled extxyz files.
        energy_ref_key: ``info``/calculator key for the reference energy.
        energy_pred_key: ``info``/calculator key for the predicted energy.
        per_atom: Use per-atom energies (eV/atom).
    """

    def __init__(
        self,
        config_files,
        *args,
        energy_ref_key: str = "REF_energy",
        energy_pred_key: str = "energy",
        per_atom: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._config_files = as_paths(config_files)
        self._energy_ref_key = energy_ref_key
        self._energy_pred_key = energy_pred_key
        self._per_atom = per_atom

    def compute_data(self, shared_ressources: dict) -> EnergyErrorData:
        import ase.io

        es_ref, es_pred = [], []
        for path in self._config_files:
            for atoms in ase.io.read(path, index=":"):
                norm = len(atoms) if self._per_atom else 1
                es_ref.append(
                    resolve_scalar(atoms, self._energy_ref_key) / norm
                )
                es_pred.append(
                    resolve_scalar(atoms, self._energy_pred_key) / norm
                )

        return EnergyErrorData(
            es_ref=np.asarray(es_ref, dtype=float).tolist(),
            es_pred=np.asarray(es_pred, dtype=float).tolist(),
            per_atom=self._per_atom,
        )
