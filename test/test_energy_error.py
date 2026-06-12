"""Tests for the energy-error (benchmark) panel."""

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("ase")
import ase.io  # noqa: E402
from ase import Atoms  # noqa: E402

from alviz import Dashboard  # noqa: E402
from alviz.panels import EnergyErrorData, ExtxyzEnergyErrorPanel  # noqa: E402


@pytest.fixture
def labelled_extxyz(tmp_path):
    rng = np.random.default_rng(0)
    configs = []
    for _ in range(7):
        n = 3
        atoms = Atoms("H" * n, positions=rng.uniform(0, 5, (n, 3)))
        e_ref = float(rng.uniform(-10, -5))
        atoms.info["REF_energy"] = e_ref
        atoms.info["energy"] = e_ref + float(rng.normal(0, 0.1))
        configs.append(atoms)
    path = tmp_path / "bench.extxyz"
    ase.io.write(path, configs)
    return path


@pytest.mark.local
def test_energy_error_end_to_end(tmp_path, labelled_extxyz):
    board = Dashboard(1, 1, fail_silently=False)
    panel = ExtxyzEnergyErrorPanel(
        config_files=labelled_extxyz, datapath=tmp_path
    )
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    assert isinstance(panel.data, EnergyErrorData)
    assert len(panel.data.es_ref) == 7
    assert len(panel.data.es_pred) == 7

    out = tmp_path / "err.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.local
def test_energy_error_zero_when_equal(tmp_path):
    atoms = Atoms("H2", positions=np.zeros((2, 3)))
    atoms.info["REF_energy"] = -4.0
    atoms.info["energy"] = -4.0
    path = tmp_path / "eq.extxyz"
    ase.io.write(path, [atoms])

    panel = ExtxyzEnergyErrorPanel(
        config_files=path, datapath=tmp_path, per_atom=False
    )
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()

    assert panel.data.es_ref == [-4.0]
    assert panel.data.es_pred == [-4.0]


@pytest.mark.local
def test_energy_error_data_rejects_length_mismatch():
    with pytest.raises(ValidationError):
        EnergyErrorData(es_ref=[1.0, 2.0], es_pred=[1.0], per_atom=True)
