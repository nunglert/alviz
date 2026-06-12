"""End-to-end test for SampledEnergyVolume against synthetic extxyz."""

import numpy as np
import pytest

pytest.importorskip("ase")
import ase.io  # noqa: E402
from ase import Atoms  # noqa: E402

from alviz import Dashboard, JsonDict  # noqa: E402
from alviz.panels import SampledEnergyVolume  # noqa: E402


@pytest.fixture
def evaluated_extxyz(tmp_path):
    rng = np.random.default_rng(0)
    configs = []
    for _ in range(8):
        n = 4
        atoms = Atoms(
            "H" * n,
            positions=rng.uniform(0, 5, size=(n, 3)),
            cell=np.eye(3) * rng.uniform(8, 12),
            pbc=True,
        )
        atoms.info["energy"] = float(rng.uniform(-10, -5))
        atoms.arrays["forces"] = rng.normal(size=(n, 3))
        configs.append(atoms)
    path = tmp_path / "evaluated.extxyz"
    ase.io.write(path, configs)
    return path


@pytest.mark.local
def test_sampled_ev_end_to_end(tmp_path, evaluated_extxyz):
    board = Dashboard(1, 1, fail_silently=False)
    panel = SampledEnergyVolume(
        config_files=evaluated_extxyz, datapath=tmp_path
    )
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    assert len(panel.data["es"]) == 8
    assert len(panel.data["vs"]) == 8
    assert len(panel.data["fs"]) == 8 * 4 * 3
    assert panel.data["per_atom"] is True

    cache = tmp_path / "sampled_energy_volume.json"
    assert cache.exists()
    assert set(JsonDict.from_json(cache).keys()) >= {"es", "fs", "vs"}

    out = tmp_path / "ev.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.local
def test_sampled_ev_per_atom_scaling(tmp_path):
    """per_atom=False leaves energy/volume untouched (total, not per atom)."""
    atoms = Atoms("H4", positions=np.zeros((4, 3)), cell=np.eye(3) * 10)
    atoms.info["energy"] = -8.0
    atoms.arrays["forces"] = np.zeros((4, 3))
    path = tmp_path / "one.extxyz"
    ase.io.write(path, [atoms])

    total = SampledEnergyVolume(
        config_files=path, datapath=tmp_path, per_atom=False
    )
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(total, ("0", "0"))
    board.allocate_panels()
    board.compute_data()

    assert total.data["es"] == [-8.0]
    # 10^3, not divided by 4 (det() carries tiny FP error).
    assert total.data["vs"][0] == pytest.approx(1000.0)
