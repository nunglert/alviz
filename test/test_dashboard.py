"""Framework-level tests for the dashboard engine.

These exercise the Entry/Dashboard lifecycle and JsonDict with a trivial
in-memory panel -- no producer artifacts or heavy dependencies required.
"""

import pytest

from alviz import Dashboard, Entry, JsonDict, PanelData, string2slice


class _SinePanel(Entry):
    """Minimal panel: computes a tiny series and plots it."""

    def allocate_panels(self, fig):
        self.ax = fig.add_subplot(self.subspec)

    def compute_data(self, shared_ressources):
        return JsonDict({"x": [0, 1, 2, 3], "y": [0, 1, 4, 9]})

    def plot_data(self):
        self.ax.plot(self.data["x"], self.data["y"])


@pytest.mark.local
def test_compute_data_type_must_match_data_class(tmp_path):
    """compute_data returning a type != data_class fails immediately."""

    class _Mismatch(Entry):
        # data_class defaults to JsonDict, but compute_data returns PanelData
        def allocate_panels(self, fig):
            self.ax = fig.add_subplot(self.subspec)

        def compute_data(self, shared_ressources):
            return PanelData()  # not a JsonDict -> invariant violation

        def plot_data(self):
            pass

    # Tested at the source (get_data); Dashboard.compute_data would re-wrap
    # this TypeError as a generic Exception under fail_silently=False.
    panel = _Mismatch(datapath=tmp_path)
    with pytest.raises(TypeError, match="data_class"):
        panel.get_data({}, force_recompute=True)


# --------------------------------------------------------------------------
# string2slice
# --------------------------------------------------------------------------
@pytest.mark.local
@pytest.mark.parametrize(
    "text, expected",
    [
        ("0", slice(0, 1, None)),
        ("2", slice(2, 3, None)),
        ("-1", slice(-1, None, None)),
        ("1:4", slice(1, 4, None)),
        (":3", slice(None, 3, None)),
        ("1:", slice(1, None, None)),
    ],
)
def test_string2slice(text, expected):
    assert string2slice(text) == expected


# --------------------------------------------------------------------------
# JsonDict
# --------------------------------------------------------------------------
@pytest.mark.local
def test_jsondict_roundtrip(tmp_path):
    d = JsonDict({"a": 1, "nested": {"b": [1, 2, 3]}})
    path = tmp_path / "data.json"
    d.to_json(path)
    loaded = JsonDict.from_json(path)
    assert loaded["a"] == 1
    assert loaded["nested"]["b"] == [1, 2, 3]


@pytest.mark.local
def test_jsondict_rejects_unserializable():
    d = JsonDict()
    with pytest.raises(ValueError):
        d["bad"] = object()


# --------------------------------------------------------------------------
# Dashboard lifecycle
# --------------------------------------------------------------------------
@pytest.mark.local
def test_dashboard_full_lifecycle(tmp_path):
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(_SinePanel(datapath=tmp_path), ("0", "0"))

    board.allocate_panels()
    board.compute_data()
    board.plot_panels()

    out = tmp_path / "board.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.local
def test_compute_data_caches_json(tmp_path):
    """compute_data writes a per-panel JSON cache keyed by entry name."""
    board = Dashboard(1, 1, fail_silently=False)
    board.register_entry(_SinePanel(datapath=tmp_path), ("0", "0"))
    board.allocate_panels()
    board.compute_data()

    cache = tmp_path / "_SinePanel.json"
    assert cache.exists()
    cached = JsonDict.from_json(cache)
    assert cached["y"] == [0, 1, 4, 9]


@pytest.mark.local
def test_register_entry_rejects_overlap(tmp_path):
    board = Dashboard(2, 2, fail_silently=False)
    board.register_entry(_SinePanel(datapath=tmp_path), ("0", "0"))
    with pytest.raises(Exception):
        # Same slot again -> must refuse.
        board.register_entry(_SinePanel(datapath=tmp_path), ("0", "0"))


@pytest.mark.local
def test_fail_silently_skips_bad_panel(tmp_path):
    """A panel that raises during compute is excluded, not fatal."""

    class _Broken(Entry):
        def allocate_panels(self, fig):
            self.ax = fig.add_subplot(self.subspec)

        def compute_data(self, shared_ressources):
            raise RuntimeError("boom")

        def plot_data(self):
            self.ax.plot([0], [0])

    board = Dashboard(1, 2, fail_silently=True)
    board.register_entry(_Broken(datapath=tmp_path, name="broken"), ("0", "0"))
    board.register_entry(_SinePanel(datapath=tmp_path), ("0", "1"))

    board.allocate_panels()
    board.compute_data()  # broken panel logged + excluded, no raise
    board.plot_panels()

    out = tmp_path / "board.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0
