"""Tests for the params (metadata) panel."""

import json

import pytest

from alviz import Dashboard, JsonDict
from alviz.panels import JsonParamsPanel


@pytest.mark.local
def test_json_params_panel(tmp_path):
    params = {"r_cut": 4.0, "n_epochs": 100, "nested": {"seed": 7}}
    pfile = tmp_path / "params.json"
    with open(pfile, "w") as f:
        json.dump(params, f)

    board = Dashboard(1, 1, fail_silently=False)
    panel = JsonParamsPanel(input_file=pfile, datapath=tmp_path, title="cfg")
    board.register_entry(panel, ("0", "0"))
    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    assert isinstance(panel.data, JsonDict)
    assert panel.data["r_cut"] == 4.0
    assert panel.data["nested"]["seed"] == 7

    out = tmp_path / "params.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0
