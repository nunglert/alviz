"""Distributions of sampled energy, force and volume.

* ``SampledEVData``              -- the data contract.
* ``SampledEnergyVolume``        -- abstract: histograms + contract.
* ``ExtxyzSampledEnergyVolume``  -- reads the distributions from extxyz.

Useful for judging the spread of a (e.g. NPT-) sampled set.
"""

from abc import abstractmethod

import numpy as np

from alviz.dashboard import Entry
from alviz.data import PanelData
from alviz.panels._atoms import as_paths, resolve_array, resolve_scalar


class SampledEVData(PanelData):
    """Per-atom (or total) energies, flattened forces, and volumes."""

    es: list[float]
    fs: list[float]
    vs: list[float]
    per_atom: bool


class SampledEnergyVolume(Entry):
    """Histograms of energy, force components and volume.

    Args (rendering):
        plot_kwargs: Extra kwargs forwarded to ``ax.hist``.
    """

    name = "sampled_energy_volume"
    data_class = SampledEVData

    def __init__(self, *args, plot_kwargs: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._plot_kwargs = plot_kwargs or {}

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 3)
        self.ax_e = fig.add_subplot(self.gs[0, 0])
        self.ax_e.set_title("Energy")
        self.ax_f = fig.add_subplot(self.gs[0, 1])
        self.ax_f.set_title("Forces")
        self.ax_v = fig.add_subplot(self.gs[0, 2])
        self.ax_v.set_title("Volume")

    @abstractmethod
    def compute_data(self, shared_ressources: dict) -> SampledEVData:
        """Produce the distributions (see :class:`SampledEVData`)."""
        raise NotImplementedError

    def _hist(self, ax, values, color, ylabel):
        kwargs = dict(
            bins=50,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            orientation="horizontal",
        )
        kwargs.update(self._plot_kwargs)
        ax.hist(values, **kwargs)
        ax.set_xlabel("Counts")
        ax.set_ylabel(ylabel)

    def plot_data(self):
        d = self.data
        e_unit = "eV/at" if d.per_atom else "eV"
        v_unit = r"$\AA^3\,\mathrm{atom}^{-1}$" if d.per_atom else r"$\AA^3$"
        self._hist(self.ax_e, d.es, "tab:blue", rf"$E$ [{e_unit}]")
        self._hist(self.ax_f, d.fs, "tab:green", r"$F$ [eV/$\AA$]")
        self._hist(self.ax_v, d.vs, "tab:red", rf"$V$ [{v_unit}]")


class ExtxyzSampledEnergyVolume(SampledEnergyVolume):
    """Read energy/force/volume distributions from extxyz files.

    Args:
        config_files: One path or list of paths to labelled extxyz files.
        energy_key: ``info``/calculator key for the energy.
        forces_key: ``arrays``/calculator key for the forces.
        per_atom: Normalize energy and volume per atom.
    """

    def __init__(
        self,
        config_files,
        *args,
        energy_key: str = "energy",
        forces_key: str = "forces",
        per_atom: bool = True,
        plot_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(*args, plot_kwargs=plot_kwargs, **kwargs)
        self._config_files = as_paths(config_files)
        self._energy_key = energy_key
        self._forces_key = forces_key
        self._per_atom = per_atom

    def compute_data(self, shared_ressources: dict) -> SampledEVData:
        import ase.io

        es, fs, vs = [], [], []
        for path in self._config_files:
            for atoms in ase.io.read(path, index=":"):
                norm = len(atoms) if self._per_atom else 1
                es.append(resolve_scalar(atoms, self._energy_key) / norm)
                fs.extend(resolve_array(atoms, self._forces_key).ravel())
                vs.append(atoms.get_volume() / norm)

        return SampledEVData(
            es=np.asarray(es, dtype=float).tolist(),
            fs=np.asarray(fs, dtype=float).tolist(),
            vs=np.asarray(vs, dtype=float).tolist(),
            per_atom=self._per_atom,
        )
