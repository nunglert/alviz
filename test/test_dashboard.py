"""Framework-level tests for the dashboard engine.

These exercise the Entry/Dashboard lifecycle and JsonDict with a trivial
in-memory panel -- no producer artifacts or heavy dependencies required.
"""

import pytest

from alviz import (
    Dashboard,
    Entry,
    JsonDict,
    PanelData,
    select_entries,
    string2slice,
)


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


def _panel(tmp_path, name):
    return _SinePanel(datapath=tmp_path, name=name)


# --------------------------------------------------------------------------
# select_entries
# --------------------------------------------------------------------------
@pytest.mark.local
def test_select_entries_only_exclude_and_order(tmp_path):
    entries = [_panel(tmp_path, n) for n in ("a", "b", "c")]
    assert [e.name for e in select_entries(entries)] == ["a", "b", "c"]
    # 'only' keeps the requested names but in the original order.
    assert [e.name for e in select_entries(entries, only=["c", "a"])] == [
        "a",
        "c",
    ]
    assert [e.name for e in select_entries(entries, exclude=["b"])] == [
        "a",
        "c",
    ]


# --------------------------------------------------------------------------
# Dashboard.from_entries / .entries
# --------------------------------------------------------------------------
@pytest.mark.local
def test_from_entries_auto_grid_and_order(tmp_path):
    entries = [_panel(tmp_path, n) for n in ("a", "b", "c")]
    board = Dashboard.from_entries(entries, ncols=2)
    assert board._ncols == 2
    assert board._nrows == 2  # ceil(3 / 2)
    assert [e.name for e in board.entries] == ["a", "b", "c"]


@pytest.mark.local
def test_from_entries_applies_selection(tmp_path):
    entries = [_panel(tmp_path, n) for n in ("a", "b", "c")]
    board = Dashboard.from_entries(entries, ncols=2, only=["b"])
    assert [e.name for e in board.entries] == ["b"]


@pytest.mark.local
def test_from_entries_colspan_spans_columns(tmp_path):
    a = _panel(tmp_path, "a")
    a.colspan = 2  # 'a' should fill the whole first row
    b = _panel(tmp_path, "b")
    board = Dashboard.from_entries([a, b], ncols=2)

    assert board._nrows == 2
    layout = board._layout
    # entry 0 ('a') occupies both columns of row 0; 'b' drops to row 1.
    assert layout[0, 0].strip() == "0" and layout[0, 1].strip() == "0"
    assert layout[1, 0].strip() == "1"


@pytest.mark.local
def test_from_entries_colspan_clamped_to_ncols(tmp_path):
    a = _panel(tmp_path, "a")
    a.colspan = 5  # wider than the grid -> clamped to ncols
    board = Dashboard.from_entries([a], ncols=2)
    assert board._nrows == 1
    assert board._layout[0, 0].strip() == "0"
    assert board._layout[0, 1].strip() == "0"


@pytest.mark.local
def test_from_entries_rowspan_fills_around(tmp_path):
    a = _panel(tmp_path, "a")
    a.rowspan = 2  # tall: occupies (0,0) and (1,0)
    b = _panel(tmp_path, "b")
    c = _panel(tmp_path, "c")
    board = Dashboard.from_entries([a, b, c], ncols=2)

    assert board._nrows == 2
    layout = board._layout
    assert layout[0, 0].strip() == "0" and layout[1, 0].strip() == "0"
    assert layout[0, 1].strip() == "1"  # 'b' to the right of 'a'
    assert layout[1, 1].strip() == "2"  # 'c' fills the hole below 'b'


# --------------------------------------------------------------------------
# Dashboard.render
# --------------------------------------------------------------------------
@pytest.mark.local
def test_render_runs_lifecycle_and_titles(tmp_path):
    board = Dashboard.from_entries([_panel(tmp_path, "a")], ncols=1)
    result = board.render(suptitle="My title")
    assert result is board  # chainable
    assert any(t.get_text() == "My title" for t in board._fig.texts)

    out = tmp_path / "board.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.local
def test_render_skips_compute_failure_at_plot(tmp_path):
    """A compute failure is shared to plot, so render stays non-fatal."""

    class _Broken(Entry):
        def allocate_panels(self, fig):
            self.ax = fig.add_subplot(self.subspec)

        def compute_data(self, shared_ressources):
            raise RuntimeError("boom")

        def plot_data(self):
            self.ax.plot([0], [0])

    board = Dashboard.from_entries(
        [_Broken(datapath=tmp_path, name="broken"), _panel(tmp_path, "ok")],
        ncols=2,
    )
    board.render()  # broken panel computed+plotted-excluded, no raise

    out = tmp_path / "board.pdf"
    board.savefig(out)
    assert out.exists() and out.stat().st_size > 0
