"""Tests for the registry-driven panel-spec machinery (alviz.spec)."""

import pytest
from alviz import Entry
from alviz.mpl_utils import JsonDict
from alviz.spec import (
    PANEL_REGISTRY,
    PanelSpec,
    build_entries,
    parse_specs,
    register_panel,
    registered_types,
    resolve_panel_list,
)
from pydantic import ValidationError

pytestmark = pytest.mark.local


# --- a throwaway panel + spec to register in each test ----------------------
class _DummyEntry(Entry):
    """Minimal concrete Entry (compute/plot are no-ops)."""

    def allocate_panels(self, fig):
        pass

    def compute_data(self, shared_ressources):
        return JsonDict({})

    def plot_data(self):
        pass


class _DummySpec(PanelSpec):
    type: str = "dummy"
    label: str = "x"
    gate: bool = True  # when False the builder returns None (absent inputs)


def _dummy_builder(spec, context, common):
    if not spec.gate:
        return None
    name = spec.name or f"dummy_{spec.label}"
    return _DummyEntry(name=name, **common)


@pytest.fixture
def registered(tmp_path):
    """Register the dummy panel; clean the registry afterwards."""
    before = dict(PANEL_REGISTRY)
    register_panel("dummy", _DummySpec, _dummy_builder)
    yield
    PANEL_REGISTRY.clear()
    PANEL_REGISTRY.update(before)


# ---------------------------------------------------------------------------
# parse_specs
# ---------------------------------------------------------------------------
def test_parse_specs_dispatches_on_type(registered):
    (spec,) = parse_specs([{"type": "dummy", "label": "a"}])
    assert isinstance(spec, _DummySpec)
    assert spec.label == "a" and spec.enabled is True


def test_parse_specs_passes_through_spec_instances(registered):
    s = _DummySpec(label="b")
    assert parse_specs([s]) == [s]


def test_parse_specs_unknown_type_raises(registered):
    with pytest.raises(ValueError, match="Unknown panel type 'nope'"):
        parse_specs([{"type": "nope"}])


def test_parse_specs_missing_type_raises(registered):
    with pytest.raises(ValueError, match="'type' key"):
        parse_specs([{"label": "a"}])


def test_parse_specs_unknown_field_is_validation_error(registered):
    # extra="forbid" -> a misspelled key is loud, not silently dropped.
    with pytest.raises(ValidationError):
        parse_specs([{"type": "dummy", "labl": "typo"}])


def test_registered_types_lists_names(registered):
    assert "dummy" in registered_types()


# ---------------------------------------------------------------------------
# resolve_panel_list
# ---------------------------------------------------------------------------
def test_resolve_defaults_disable_add():
    d1 = _DummySpec(label="1")
    d2 = _DummySpec(label="2", name="keep")
    extra = _DummySpec(label="3")
    out = resolve_panel_list([d1, d2], add=[extra], disable=["keep"])
    assert out == [d1, extra]  # d2 dropped by its name, extra appended last


def test_resolve_disable_by_type_drops_all_of_type():
    out = resolve_panel_list(
        [_DummySpec(label="1"), _DummySpec(label="2")], disable=["dummy"]
    )
    assert out == []


def test_resolve_panels_is_authoritative():
    default = _DummySpec(label="d")
    override = _DummySpec(label="o")
    out = resolve_panel_list([default], panels=[override], add=[default])
    assert out == [override]  # defaults + add ignored when panels given


# ---------------------------------------------------------------------------
# build_entries
# ---------------------------------------------------------------------------
def test_build_entries_skips_disabled_and_absent(registered, tmp_path):
    specs = [
        _DummySpec(label="on"),
        _DummySpec(label="off", enabled=False),
        _DummySpec(label="absent", gate=False),  # builder returns None
    ]
    entries = build_entries(
        specs, context=None, datapath=tmp_path, force_recompute=False
    )
    assert [e.name for e in entries] == ["dummy_on"]


def test_build_entries_duplicate_name_raises(registered, tmp_path):
    specs = [_DummySpec(label="same"), _DummySpec(label="same")]
    with pytest.raises(ValueError, match="same name"):
        build_entries(
            specs, context=None, datapath=tmp_path, force_recompute=False
        )


def test_build_entries_forwards_common(registered, tmp_path):
    (entry,) = build_entries(
        [_DummySpec(label="z")],
        context=None,
        datapath=tmp_path,
        force_recompute=True,
    )
    assert entry.datapath == tmp_path
    assert entry._force_recompute is True
