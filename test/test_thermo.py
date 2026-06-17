"""Tests for the thermodynamic heatmap panel (abstract base + contract).

No Extxyz reader exists (the grid comes from postprocessing), so a tiny
in-memory subclass stands in for a project's compute_data.
"""

import pytest
from pydantic import ValidationError

from alviz import Dashboard
from alviz.panels import HeatmapData, HeatmapPanel


class _InMemoryHeatmap(HeatmapPanel):
    def compute_data(self, shared_ressources):
        return HeatmapData(
            t=[500.0, 1000.0, 1500.0],  # 3 temperatures
            p=[0.0, 1.0],  # 2 pressures
            obs=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # (len(p), len(t))
            quantity="heatcapacity",
        )


@pytest.mark.local
def test_heatmap_end_to_end(tmp_path):
    board = Dashboard(1, 1, fail_silently=False)
    panel = _InMemoryHeatmap(datapath=tmp_path)
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    assert isinstance(panel.data, HeatmapData)
    assert panel.data.quantity == "heatcapacity"

    cache = tmp_path / "heatmap.json"
    assert cache.exists()
    assert HeatmapData.from_json(cache).obs == panel.data.obs

    out = tmp_path / "heatmap.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.local
def test_heatmap_data_rejects_bad_shape():
    with pytest.raises(ValidationError):
        # obs has 2 rows but p has 3 -> shape mismatch
        HeatmapData(t=[1.0, 2.0], p=[0.0, 1.0, 2.0], obs=[[1.0, 2.0]])
