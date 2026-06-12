"""Typed, validated data contracts for panels.

A panel's ``compute_data`` returns a ``PanelData`` subclass that declares
exactly the fields its ``plot_data`` consumes. pydantic validates types,
required fields, and (via validators) cross-field invariants at construction,
so a wrong-shaped result fails immediately with a clear message rather than
as a cryptic error mid-plot. Instances are immutable and JSON-round-trip for
the dashboard's per-panel cache.

This is *the* interface between a panel's data production (downstream,
project-owned) and its rendering (here, in alviz): subclass ``PanelData`` to
define what a panel needs, implement ``compute_data`` to produce it.
"""

from pydantic import BaseModel, ConfigDict


class PanelData(BaseModel):
    """Base for typed panel-data contracts (immutable, JSON-serializable)."""

    model_config = ConfigDict(frozen=True)

    def to_json(self, path, indent=None):
        """Write the validated data to ``path`` as JSON (cache)."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=indent))

    @classmethod
    def from_json(cls, path):
        """Load + re-validate cached data from ``path``."""
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())
