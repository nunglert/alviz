"""Tests for the parity panel: the extxyz reader, the data contract, and a
custom (non-extxyz) compute_data implementation.
"""

import numpy as np
import pytest
from pydantic import ValidationError

ase = pytest.importorskip("ase")
ase_io = pytest.importorskip("ase.io")

from ase import Atoms  # noqa: E402

from alviz import Dashboard  # noqa: E402
from alviz.panels import (  # noqa: E402
    ExtxyzParityPanel,
    ParityData,
    ParityPanel,
)


def _make_config(rng, n_atoms, e_pred_offset, f_noise):
    """One H_n config with reference + predicted energy/forces labels."""
    atoms = Atoms(
        "H" * n_atoms,
        positions=rng.uniform(0, 5, size=(n_atoms, 3)),
        cell=np.eye(3) * 10,
        pbc=True,
    )
    e_ref = float(rng.uniform(-10, -5))
    f_ref = rng.normal(size=(n_atoms, 3))
    atoms.info["REF_energy"] = e_ref
    atoms.arrays["REF_forces"] = f_ref
    atoms.info["energy"] = e_ref + e_pred_offset
    atoms.arrays["forces"] = f_ref + f_noise * rng.normal(size=(n_atoms, 3))
    return atoms


@pytest.fixture
def labelled_extxyz(tmp_path):
    rng = np.random.default_rng(0)
    configs = [
        _make_config(rng, n_atoms=4, e_pred_offset=0.1, f_noise=0.05)
        for _ in range(6)
    ]
    path = tmp_path / "evaluated.extxyz"
    ase_io.write(path, configs)
    return path


# --------------------------------------------------------------------------
# ExtxyzParityPanel (the batteries-included reader)
# --------------------------------------------------------------------------
@pytest.mark.local
def test_extxyz_parity_end_to_end(tmp_path, labelled_extxyz):
    board = Dashboard(1, 1, fail_silently=False)
    panel = ExtxyzParityPanel(config_files=labelled_extxyz, datapath=tmp_path)
    board.register_entry(panel, ("0", "0"))

    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    # data is a validated ParityData (attribute access).
    assert isinstance(panel.data, ParityData)
    assert len(panel.data.e_obs) == 6
    assert panel.data.e_rmse >= 0
    assert len(panel.data.f_obs) == 6 * 4 * 3

    cache = tmp_path / "parity.json"
    assert cache.exists()
    # cache reloads + re-validates through the contract.
    assert ParityData.from_json(cache).e_obs == panel.data.e_obs

    out = tmp_path / "parity.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.local
def test_extxyz_parity_perfect_prediction(tmp_path):
    rng = np.random.default_rng(1)
    atoms = Atoms(
        "H3", positions=rng.uniform(0, 5, (3, 3)), cell=np.eye(3) * 10
    )
    e, f = -7.5, rng.normal(size=(3, 3))
    atoms.info["REF_energy"] = e
    atoms.info["energy"] = e
    atoms.arrays["REF_forces"] = f
    atoms.arrays["forces"] = f.copy()
    path = tmp_path / "perfect.extxyz"
    ase_io.write(path, [atoms])

    panel = ExtxyzParityPanel(config_files=path, datapath=tmp_path)
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()

    assert panel.data.e_rmse == pytest.approx(0.0)
    assert panel.data.f_rmse == pytest.approx(0.0)


@pytest.mark.local
def test_extxyz_parity_energy_only(tmp_path, labelled_extxyz):
    panel = ExtxyzParityPanel(
        config_files=labelled_extxyz, datapath=tmp_path, show_forces=False
    )
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    assert panel.data.f_obs is None
    assert panel.ax_f is None


@pytest.mark.local
def test_extxyz_parity_custom_keys(tmp_path):
    rng = np.random.default_rng(2)
    atoms = Atoms(
        "H2", positions=rng.uniform(0, 5, (2, 3)), cell=np.eye(3) * 10
    )
    atoms.info["dft_E"] = -3.0
    atoms.info["mace_E"] = -2.7
    atoms.arrays["dft_F"] = rng.normal(size=(2, 3))
    atoms.arrays["mace_F"] = rng.normal(size=(2, 3))
    path = tmp_path / "custom.extxyz"
    ase_io.write(path, [atoms])

    panel = ExtxyzParityPanel(
        config_files=path,
        datapath=tmp_path,
        energy_ref_key="dft_E",
        energy_pred_key="mace_E",
        forces_ref_key="dft_F",
        forces_pred_key="mace_F",
        per_atom=False,
    )
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()

    assert panel.data.per_atom is False
    assert panel.data.e_obs == [-3.0]
    assert panel.data.e_pred == [-2.7]


# --------------------------------------------------------------------------
# The data contract enforces shape (pydantic)
# --------------------------------------------------------------------------
@pytest.mark.local
def test_parity_data_rejects_missing_field():
    with pytest.raises(ValidationError):
        ParityData(e_obs=[1.0], e_pred=[1.0], e_mae=0.0, per_atom=True)


@pytest.mark.local
def test_parity_data_rejects_length_mismatch():
    with pytest.raises(ValidationError):
        ParityData(
            e_obs=[1.0, 2.0],
            e_pred=[1.0],
            e_rmse=0.0,
            e_mae=0.0,
            per_atom=True,
        )


@pytest.mark.local
def test_parity_data_coerces_types():
    # pydantic coerces int -> float per the contract.
    d = ParityData(
        e_obs=[1, 2], e_pred=[1, 2], e_rmse=0, e_mae=0, per_atom=True
    )
    assert d.e_obs == [1.0, 2.0]
    assert isinstance(d.e_rmse, float)


# --------------------------------------------------------------------------
# A custom data source: subclass ParityPanel, no extxyz involved
# --------------------------------------------------------------------------
@pytest.mark.local
def test_custom_parity_source(tmp_path):
    class InMemoryParity(ParityPanel):
        def compute_data(self, shared_ressources):
            return ParityData(
                e_obs=[-1.0, -2.0, -3.0],
                e_pred=[-1.1, -1.9, -3.2],
                e_rmse=0.15,
                e_mae=0.13,
                per_atom=True,
            )

    panel = InMemoryParity(datapath=tmp_path, show_forces=False)
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    assert isinstance(panel.data, ParityData)
    assert panel.data.e_obs == [-1.0, -2.0, -3.0]
    out = tmp_path / "custom.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0
