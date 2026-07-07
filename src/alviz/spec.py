"""Panel specs: a typed, registry-driven description of a dashboard's panels.

A dashboard can be described as *data* -- an ordered list of :class:`PanelSpec`,
each a pydantic model naming a panel ``type`` plus that panel's parameters --
instead of hand-wired :class:`~alviz.dashboard.Entry` construction. A registry
maps each ``type`` to its spec model and a *builder*. The builder turns a
validated spec + an opaque, caller-supplied ``context`` (whatever object holds
"where the files live") into an ``Entry``, or ``None`` when the panel's inputs
are absent -- so panel *selection* stays data-driven, exactly as hand-wired
``assemble``-style code already gates on file existence.

This is the domain-agnostic machinery. Concrete panel types -- their spec
fields and builders -- are registered by the consuming project via
:func:`register_panel`, keeping alviz free of any producer knowledge (it has
more than one consumer). ``context`` is forwarded to builders verbatim and is
never inspected here.

Typical use by a consumer::

    class SteinhardtSpec(PanelSpec):
        type: str = "steinhardt"
        r_cut: float
        l: int = 6

    def _build(spec, ctx, common):
        if not ctx.ns_dirs:
            return None
        return SteinhardtPanel(result_dirs=ctx.ns_dirs, r_cut=spec.r_cut,
                               l=spec.l, name=f"steinhardt_q{spec.l}", **common)

    register_panel("steinhardt", SteinhardtSpec, _build)

    specs = resolve_panel_list(default_specs, add=parse_specs(cfg["add"]))
    entries = build_entries(specs, ctx, datapath=dp, force_recompute=False)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict, create_model


class PanelSpec(BaseModel):
    """Base for a typed panel description (one entry in a dashboard's list).

    Subclasses add the panel's parameters as fields (with ``Field(...,
    description=...)`` so a config scaffolder can document them). ``extra`` is
    forbidden, so an unknown/misspelled key is a validation error rather than a
    silently ignored setting.

    Attributes:
        type: The registry key selecting this panel's model + builder.
            Subclasses should default it to their registered name.
        enabled: When False, :func:`build_entries` skips the panel (a config
            can turn a curated default off without removing it).
        name: Optional explicit entry name (else the builder decides). Handy to
            disambiguate several instances of one ``type``.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    enabled: bool = True
    name: Optional[str] = None


# type name -> (spec model, builder). ``builder(spec, context, common) ->
# Entry | None``; ``common`` is ``{"datapath", "force_recompute"}``.
@dataclass
class _Registered:
    spec_model: type
    builder: Callable


PANEL_REGISTRY: dict = {}


def register_panel(type_name: str, spec_model, builder) -> None:
    """Register a panel ``type``: its :class:`PanelSpec` model + builder.

    ``builder(spec, context, common) -> Entry | None`` receives the validated
    spec, the opaque caller context, and ``common`` (``datapath`` +
    ``force_recompute`` kwargs shared by every ``Entry``); it returns ``None``
    when the panel's inputs are absent. Re-registering a name overwrites (so a
    module re-import is harmless).
    """
    PANEL_REGISTRY[type_name] = _Registered(spec_model, builder)


def register_simple(type_name: str, builder) -> type:
    """Register a *param-less* panel type: a trivial spec + a simple builder.

    Synthesizes a :class:`PanelSpec` subclass with no extra fields (so the panel
    is still describable/selectable in config) and registers it. ``builder`` is
    the convenience form ``builder(context, common) -> Entry | None`` -- it does
    not receive the spec (there are no params to read). For a panel *with*
    parameters, define a :class:`PanelSpec` subclass and use
    :func:`register_panel` directly. Returns the synthesized spec model.
    """
    model = create_model(
        f"{type_name}_spec", __base__=PanelSpec, type=(str, type_name)
    )
    register_panel(
        type_name, model, lambda spec, ctx, common: builder(ctx, common)
    )
    return model


def registered_types() -> list:
    """The registered panel ``type`` names, sorted (for config scaffolding)."""
    return sorted(PANEL_REGISTRY)


def panel_spec_model(type_name: str):
    """The :class:`PanelSpec` subclass registered for ``type_name``."""
    return PANEL_REGISTRY[type_name].spec_model


def parse_specs(raw) -> list:
    """Validate raw panel dicts into their registered :class:`PanelSpec`s.

    Each item is dispatched on its ``type`` to the registered model and
    validated (unknown fields raise -- see ``extra="forbid"``). Items already
    a :class:`PanelSpec` pass through. Raises ``ValueError`` on a missing/
    unknown ``type``.
    """
    specs = []
    for i, item in enumerate(raw or []):
        if isinstance(item, PanelSpec):
            specs.append(item)
            continue
        if not isinstance(item, dict) or "type" not in item:
            raise ValueError(
                f"Panel spec #{i} must be a mapping with a 'type' key, "
                f"got {item!r}."
            )
        t = item["type"]
        reg = PANEL_REGISTRY.get(t)
        if reg is None:
            raise ValueError(
                f"Unknown panel type {t!r}; registered: {registered_types()}."
            )
        specs.append(reg.spec_model.model_validate(item))
    return specs


def apply_globals(spec, globals: dict):
    """Fill a spec's *unset* fields from ``globals`` of the same name.

    A field the user did not explicitly set (not in ``model_fields_set``) whose
    name matches a key in ``globals`` takes the global value; an explicitly-set
    field (a per-panel override) is left untouched. Fields with no matching
    global keep their own default. Returns ``spec`` unchanged if nothing fills,
    else a copy. This is what makes global settings inherit into every panel
    automatically -- a panel only opts in by *declaring* the field.
    """
    fill = {
        name: globals[name]
        for name in type(spec).model_fields
        if name not in spec.model_fields_set and name in globals
    }
    return spec.model_copy(update=fill) if fill else spec


def _disabled(spec, disable: set) -> bool:
    """A spec is disabled if its explicit ``name`` or its ``type`` is listed."""
    return (spec.name is not None and spec.name in disable) or (
        spec.type in disable
    )


def resolve_panel_list(
    defaults, *, panels=None, add=None, disable=None
) -> list:
    """Merge a curated default spec list with a config's overrides.

    * ``panels`` -- if given, it is *fully authoritative*: the returned list is
      exactly it, and ``defaults`` / ``disable`` / ``add`` are ignored (you've
      already listed precisely what you want -- the power-user escape hatch).
    * otherwise start from ``defaults``, drop any whose ``name``/``type`` is in
      ``disable``, then append ``add`` (order preserved -- extras render last).

    All arguments are lists of :class:`PanelSpec` (``disable`` is a list of
    names/types). Returns a new list.
    """
    if panels is not None:
        return list(panels)
    base = list(defaults)
    if disable:
        drop = set(disable)
        base = [s for s in base if not _disabled(s, drop)]
    return base + list(add or [])


def build_entries(specs, context, *, datapath, force_recompute: bool) -> list:
    """Build the entries for a resolved spec list against a ``context``.

    Skips disabled specs, calls each spec's registered builder with
    ``(spec, context, common)`` where ``common = {datapath, force_recompute}``,
    and drops the ``None`` results (absent inputs). Raises ``ValueError`` if two
    built entries resolve to the same ``name`` -- entries are cached and laid
    out by name, so a collision would clobber a sibling's cache; give repeated
    panels distinct names (e.g. keyed by their params).
    """
    common = {"datapath": datapath, "force_recompute": force_recompute}
    entries, seen = [], set()
    for spec in specs:
        if not spec.enabled:
            continue
        reg = PANEL_REGISTRY.get(spec.type)
        if reg is None:
            raise ValueError(
                f"Unknown panel type {spec.type!r}; registered: "
                f"{registered_types()}."
            )
        entry = reg.builder(spec, context, common)
        if entry is None:
            continue
        if entry.name in seen:
            raise ValueError(
                f"Two panels resolved to the same name {entry.name!r}; give "
                "repeated panels distinct names (entries cache/lay out by "
                "name)."
            )
        seen.add(entry.name)
        entries.append(entry)
    return entries
