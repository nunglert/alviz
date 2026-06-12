"""End-to-end test for ParityPanel against synthetic extxyz artifacts.

Builds a small extxyz file carrying both reference (REF_energy/REF_forces)
and predicted (energy/forces) labels -- the documented panel contract --
then drives it through the full Dashboard lifecycle. No cluster, no model,
no producer package required.
"""

import numpy as np
import pytest

ase = pytest.importorskip("ase")
ase_io = pytest.importorskip("ase.io")

from ase import Atoms

from alviz import Dashboard, JsonDict
from alviz.panels import ParityPanel


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

    # Reference labels (ground truth), as remote_handler writes them.
    atoms.info["REF_energy"] = e_ref
    atoms.arrays["REF_forces"] = f_ref

    # Predicted labels (model), in the ASE calculator-default keys. A small
    # offset/noise so RMSE/MAE are non-zero but finite.
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


@pytest.mark.local
def test_parity_panel_end_to_end(tmp_path, labelled_extxyz):
    board = Dashboard(1, 1, fail_silently=False)
    panel = ParityPanel(config_files=labelled_extxyz, datapath=tmp_path)
    board.register_entry(panel, ("0", "0"))

    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    # Data was computed, cached, and is sane.
    assert panel.data["n_configs"] == 6
    assert panel.data["e_rmse"] >= 0
    assert panel.data["f_rmse"] >= 0
    assert len(panel.data["e_obs"]) == 6
    # 6 configs * 4 atoms * 3 components.
    assert len(panel.data["f_obs"]) == 6 * 4 * 3

    cache = tmp_path / "parity.json"
    assert cache.exists()
    assert "e_mae" in JsonDict.from_json(cache)

    out = tmp_path / "parity.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.local
def test_parity_perfect_prediction_is_zero_error(tmp_path):
    """If predicted == reference, RMSE/MAE collapse to zero."""
    rng = np.random.default_rng(1)
    atoms = Atoms(
        "H3",
        positions=rng.uniform(0, 5, size=(3, 3)),
        cell=np.eye(3) * 10,
        pbc=True,
    )
    e = -7.5
    f = rng.normal(size=(3, 3))
    atoms.info["REF_energy"] = e
    atoms.info["energy"] = e
    atoms.arrays["REF_forces"] = f
    atoms.arrays["forces"] = f.copy()

    path = tmp_path / "perfect.extxyz"
    ase_io.write(path, [atoms])

    panel = ParityPanel(config_files=path, datapath=tmp_path)
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()

    assert panel.data["e_rmse"] == pytest.approx(0.0)
    assert panel.data["e_mae"] == pytest.approx(0.0)
    assert panel.data["f_rmse"] == pytest.approx(0.0)


@pytest.mark.local
def test_parity_energy_only(tmp_path, labelled_extxyz):
    """show_forces=False renders a single-axis parity with no force data."""
    panel = ParityPanel(
        config_files=labelled_extxyz,
        datapath=tmp_path,
        show_forces=False,
    )
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    assert "f_obs" not in panel.data
    assert panel.ax_f is None


@pytest.mark.local
def test_parity_custom_keys(tmp_path):
    """Reference/predicted keys are fully configurable."""
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

    panel = ParityPanel(
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

    assert panel.data["per_atom"] is False
    assert panel.data["e_obs"] == [-3.0]
    assert panel.data["e_pred"] == [-2.7]
