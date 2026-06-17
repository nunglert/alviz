"""Tests for the ensemble-uncertainty panel."""

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("ase")
import ase.io  # noqa: E402
from ase import Atoms  # noqa: E402

from alviz import Dashboard  # noqa: E402
from alviz.panels import (  # noqa: E402
    ExtxyzSampleUncertainties,
    UncertaintyData,
)
from alviz.panels.nested_sampling.uncertainty import _pad_to_grid  # noqa: E402


def _iteration_file(tmp_path, name, n_configs, rng):
    configs = []
    for _ in range(n_configs):
        atoms = Atoms("H2", positions=rng.uniform(0, 5, (2, 3)))
        atoms.info["energy_uncert"] = float(rng.uniform(0, 0.5))
        atoms.info["forces_uncert_normed"] = float(rng.uniform(0, 0.5))
        configs.append(atoms)
    path = tmp_path / name
    ase.io.write(path, configs)
    return path


@pytest.mark.local
def test_uncertainties_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    files = [
        _iteration_file(tmp_path, f"ns.{i}.extxyz", 5, rng) for i in range(3)
    ]

    board = Dashboard(1, 1, fail_silently=False)
    panel = ExtxyzSampleUncertainties(traj_files=files, datapath=tmp_path)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    assert isinstance(panel.data, UncertaintyData)
    assert panel.data.n_iterations == 3
    assert len(panel.data.euncs) == 3
    assert all(len(row) == 5 for row in panel.data.euncs)

    out = tmp_path / "uncert.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.local
def test_uncertainties_ragged_iterations(tmp_path):
    """Iterations with different config counts still render (NaN-padded)."""
    rng = np.random.default_rng(1)
    files = [
        _iteration_file(tmp_path, "ns.0.extxyz", 3, rng),
        _iteration_file(tmp_path, "ns.1.extxyz", 6, rng),
    ]
    board = Dashboard(1, 1, fail_silently=False)
    panel = ExtxyzSampleUncertainties(traj_files=files, datapath=tmp_path)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()  # must not raise on ragged rows

    out = tmp_path / "ragged.pdf"
    board.savefig(out)
    assert out.exists()


@pytest.mark.local
def test_pad_to_grid_pads_with_nan():
    # Row 0 has 3 entries, row 1 has 1 -> pad row 1 to width 3 with NaN.
    grid = _pad_to_grid([[1.0, 2.0, 9.0], [3.0]])
    assert grid.shape == (2, 3)
    assert np.isnan(grid[1, 1]) and np.isnan(grid[1, 2])
    assert grid[0, 0] == 1.0 and grid[1, 0] == 3.0


@pytest.mark.local
def test_uncertainties_custom_keys(tmp_path):
    rng = np.random.default_rng(2)
    atoms = Atoms("H2", positions=rng.uniform(0, 5, (2, 3)))
    atoms.info["e_sig"] = 0.1
    atoms.info["f_sig"] = 0.2
    path = tmp_path / "ns.0.extxyz"
    ase.io.write(path, [atoms])

    panel = ExtxyzSampleUncertainties(
        traj_files=[path],
        datapath=tmp_path,
        energy_uncert_key="e_sig",
        forces_uncert_key="f_sig",
    )
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()

    assert panel.data.euncs == [[0.1]]
    assert panel.data.funcs == [[0.2]]


@pytest.mark.local
def test_uncertainty_data_requires_fields():
    with pytest.raises(ValidationError):
        UncertaintyData(euncs=[[0.1]], funcs=[[0.2]])  # missing n_iterations
