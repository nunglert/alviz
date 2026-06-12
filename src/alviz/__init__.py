"""alviz -- grid-dashboard visualization for active-learning MLFF workflows.

Public API:
    * :class:`~alviz.dashboard.Dashboard` -- grid layout + lifecycle.
    * :class:`~alviz.dashboard.Entry` -- abstract panel base class.
    * :class:`~alviz.mpl_utils.JsonDict` -- JSON-backed panel-data container.

alviz is a *consumer* of producer artifacts: panels read files with a
documented schema and never import the internals of the packages that
produced them (keeping dependencies one-way).
"""

from alviz.dashboard import Dashboard, Entry, string2slice
from alviz.data import PanelData
from alviz.mpl_utils import JsonDict

__all__ = ["Dashboard", "Entry", "string2slice", "JsonDict", "PanelData"]
__version__ = "0.1.0"
