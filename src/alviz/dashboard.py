"""The dashboard framework: ``Entry`` panels laid out by a ``Dashboard``.

This is the domain-agnostic engine. An :class:`Entry` is a self-contained
panel that knows how to (1) allocate its subplot(s), (2) compute its data
into a :class:`~alviz.mpl_utils.JsonDict`, and (3) plot that data. A
:class:`Dashboard` arranges entries on a grid and drives the
compute/plot/save lifecycle, with per-panel JSON caching so expensive
computations are not repeated.

Concrete panels live in ``alviz.panels`` and couple to *producer artifacts*
(files with a documented schema) -- never to a producer package's internals.
"""

import logging
import os
import traceback
from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, SubplotSpec

from alviz.data import PanelData
from alviz.mpl_utils import JsonDict

logger = logging.getLogger(__name__)


def select_entries(entries, only=None, exclude=None) -> list:
    """Filter entries by their ``name`` attribute, preserving order.

    Args:
        entries: Iterable of :class:`Entry`.
        only: If given, keep only entries whose name is in this collection.
        exclude: Entry names to drop.

    A generic helper -- selecting which panels to show is dashboard logic, not
    anything a particular data producer needs to know about.
    """
    only = set(only) if only else None
    exclude = set(exclude or ())
    return [
        e
        for e in entries
        if (only is None or e.name in only) and e.name not in exclude
    ]


def _grid_placements(entries, ncols: int) -> list:
    """First-fit packing of entries (with spans) into an ``ncols``-wide grid.

    Each entry may carry ``rowspan``/``colspan`` attributes (default 1);
    ``colspan`` is clamped to ``ncols``. Returns a list of
    ``(entry, row, col, rowspan, colspan)`` placed row-major, top-left-first,
    filling holes left by wider/taller neighbours. Rows grow as needed.
    """
    occupied: set = set()
    placements = []

    def fits(r, c, rs, cs):
        return all(
            (r + i, c + j) not in occupied
            for i in range(rs)
            for j in range(cs)
        )

    for entry in entries:
        rs = max(1, int(getattr(entry, "rowspan", 1)))
        cs = min(ncols, max(1, int(getattr(entry, "colspan", 1))))
        r = 0
        while True:
            col = next(
                (c for c in range(ncols - cs + 1) if fits(r, c, rs, cs)),
                None,
            )
            if col is not None:
                for i in range(rs):
                    for j in range(cs):
                        occupied.add((r + i, col + j))
                placements.append((entry, r, col, rs, cs))
                break
            r += 1

    return placements


def string2slice(string: str):
    """Convert an index string to an ``int``-like or ``slice``.

    ``"2"`` -> ``slice(2, 3)``, ``"-1"`` -> ``slice(-1, None)``,
    ``"1:4"`` -> ``slice(1, 4)``.
    """
    if ":" not in string:
        try:
            i = int(string)
            j = i + 1
            if i == -1:
                j = None
            return slice(i, j, None)
        except ValueError:
            pass
    i = []
    for s in string.split(":"):
        if s == "":
            i.append(None)
        else:
            i.append(int(s))
    i += (3 - len(i)) * [None]
    return slice(*i)


class Entry(ABC):
    """Abstract base class for a dashboard entry (panel).

    Each entry computes its own data and plots it.

    Args:
        datapath: Path to store/load the computed data.
        force_recompute: Whether to force recomputation of data.
        indent: Indentation level for JSON serialization.
        name: Name of the entry. If None, the class name is used.

    Methods to implement:
        allocate_panels(self, fig): Allocate subplots in the figure.
        compute_data(self, shared_ressources): Compute and return the panel's
            data -- a ``JsonDict`` (free-form) or a ``PanelData`` subclass
            (typed/validated). Set ``data_class`` to the latter so the cache
            reloads (and re-validates) it.
        plot_data(self): Plot the computed data on the allocated subplots.
    """

    # How a cached result is reloaded. JsonDict by default (free-form);
    # typed panels override with their PanelData subclass.
    data_class = JsonDict

    def __init__(
        self,
        datapath,
        force_recompute: bool = False,
        indent: Optional[int] = None,
        name: Optional[str] = None,
    ):
        assert os.path.exists(datapath), "datapath needs to exist!"
        self.datapath = datapath
        self._force_recompute = force_recompute
        self._indent = indent

        self._subspec = None
        self._data = None
        self._fig = None

        # Name resolution: explicit argument > class-level ``name`` attribute
        # > the class name. The class-attribute case is left to normal
        # attribute lookup so a subclass's declared ``name`` is not shadowed.
        if name is not None:
            self.name = name
        elif not hasattr(type(self), "name"):
            self.name = type(self).__name__

    def get_data(self, shared_ressources, force_recompute) -> None:
        """Get or compute the data for this entry, caching it as JSON.

        Args:
            shared_ressources: Shared ressources among entries.
            force_recompute: Whether to force recomputation of data.
        """
        # File to store/load the data
        json_file = self.datapath / f"{self.name}.json"
        # Check for recomputation flag in the entry or globally
        force_recomp = force_recompute or self._force_recompute
        # Load from file if it exists and no recomputation is forced
        if os.path.exists(json_file) and not force_recomp:
            logger.info(f"... Loading from file: {self.name}")
            self.data = self.data_class.from_json(json_file)
        else:
            logger.info(f"... Computing: {self.name}")
            # Compute the data
            data = self.compute_data(shared_ressources)
            # Enforce the panel invariant: compute_data's type must match
            # data_class, so the JSON cache reloads as the same type. Caught
            # now (fresh compute) rather than on the next run's cache hit.
            if not isinstance(data, self.data_class):
                raise TypeError(
                    f"{type(self).__name__}.compute_data returned "
                    f"{type(data).__name__}, but data_class is "
                    f"{self.data_class.__name__}; set data_class to the type "
                    "compute_data returns so the cache round-trips."
                )
            self.data = data
            self.data.to_json(json_file, indent=self._indent)

    def register_arguments(self, parser):
        pass

    @abstractmethod
    def allocate_panels(self, fig):
        """Allocate subplots in the given figure.

        Args:
            fig (plt.Figure): The figure to allocate subplots in.
        """
        return

    @abstractmethod
    def compute_data(self, shared_ressources: dict):
        """Compute and return the panel's data.

        Return a ``JsonDict`` (free-form) or a ``PanelData`` subclass
        (typed/validated). Whatever type this returns must match the panel's
        ``data_class`` so the JSON cache reloads correctly.
        """
        return

    @abstractmethod
    def plot_data(self):
        """Plot the computed data on the allocated subplots."""
        return

    def plot_panel(self):
        """Plot the panel after ensuring it was registered and computed."""
        assert self.subspec is not None, "Entry needs to be registered!"
        assert self.data is not None, "Data needs to be computed!"
        self.plot_data()

    @property
    def subspec(self):
        return self._subspec

    @subspec.setter
    def subspec(self, new_value: SubplotSpec):
        self._subspec = new_value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        assert isinstance(
            val, (JsonDict, PanelData)
        ), "data must be a JsonDict or a PanelData"
        self._data = val

    @classmethod
    def as_standalone(cls, width, height, **kwargs):
        """Build a 1x1 dashboard holding a single entry of this class."""
        board = Dashboard(1, 1, width, height)
        board.register_entry(cls(**kwargs), ("0", "0"))
        board.allocate_panels()
        board.compute_data()
        board.plot_panels()
        return board


class Dashboard:
    """A grid of :class:`Entry` panels with shared compute/plot lifecycle.

    Each entry occupies one or more slots in the grid and owns its data
    computation and plotting.

    Args:
        nrows: Number of rows in the grid.
        ncols: Number of columns in the grid.
        width_per_panel: Width of each panel.
        height_per_panel: Height of each panel.
        fail_silently: If True, log and continue on per-panel errors rather
            than raising.
    """

    def __init__(
        self,
        nrows: int,
        ncols: int,
        width_per_panel: float = 4,
        height_per_panel: float = 4,
        fail_silently: bool = True,
    ):
        self._nrows = nrows
        self._ncols = ncols
        self._width_per_panel = width_per_panel
        self._fail_silently = fail_silently
        self._global_args = None
        self._parser = None

        self._entries: list[Entry] = []
        self._shared_ressources = {}

        # Layout management
        self._layout = np.zeros((nrows, ncols), dtype="object")
        self._layout[...] = "___"
        self._empty_layout = self._layout.copy()
        self._index_array = np.arange(nrows * ncols).reshape(nrows, ncols)

        # Constrained layout (not tight_layout) so spacing is computed
        # automatically *including inside subgridspecs* -- panels with several
        # sub-axes get spaced without per-panel wspace tuning.
        self._fig = plt.figure(
            figsize=(width_per_panel * ncols, height_per_panel * nrows),
            layout="constrained",
        )

        # Handles the layout of the figure
        self._gs = GridSpec(
            nrows,
            ncols,
            figure=self._fig,
            height_ratios=[1.0] * nrows,
        )

    @classmethod
    def from_entries(
        cls,
        entries,
        *,
        ncols: int = 2,
        width_per_panel: float = 4.0,
        height_per_panel: float = 4.0,
        only=None,
        exclude=None,
        fail_silently: bool = True,
    ) -> "Dashboard":
        """Build a board from a flat list of entries on an auto-sized grid.

        Entries are packed row-major; rows grow to fit. A panel may occupy
        more than one cell by declaring a ``colspan`` and/or ``rowspan``
        attribute (default 1 each) -- e.g. ``class WidePanel(Entry): colspan
        = 2``, or ``panel.colspan = 2`` on an instance. ``colspan`` is clamped
        to ``ncols``. ``only``/``exclude`` filter by name first (see
        :func:`select_entries`). Callers needing explicit coordinates can still
        construct ``Dashboard`` directly and call :meth:`register_entry`.
        """
        entries = select_entries(entries, only=only, exclude=exclude)
        placements = _grid_placements(entries, ncols)
        nrows = max((r + rs for _, r, _, rs, _ in placements), default=1)
        board = cls(
            nrows,
            ncols,
            width_per_panel,
            height_per_panel,
            fail_silently=fail_silently,
        )
        for entry, r, c, rs, cs in placements:
            board.register_entry(entry, (f"{r}:{r + rs}", f"{c}:{c + cs}"))
        return board

    @property
    def entries(self) -> list:
        """The registered entries, in registration (grid) order."""
        return list(self._entries)

    def register_entry(self, entry: Entry, coordinates: tuple[str, str]):
        """Register an entry at the given grid coordinates.

        Args:
            entry: The entry to register (None is skipped).
            coordinates: ``("row_start:row_end", "col_start:col_end")`` with
                inclusive starts and exclusive ends.
        """
        if entry is None:
            logger.info("Skipping ..")
            return

        # Get the coordinates of the requested slot
        xslice = string2slice(coordinates[0])
        yslice = string2slice(coordinates[1])
        # Check if the slot is free
        if np.sum(self._layout[xslice, yslice] != "___") != 0:
            target_alloc = self._empty_layout.copy()
            target_alloc[xslice, yslice] = f"{' X '}"
            raise Exception(
                "The slot is already blocked. You tried to allocate \n"
                f"{target_alloc}\n"
                ", but the following "
                "occupation was found: \n"
                f"{self._layout}"
            )
        # Register the entry named as the next index
        self._layout[xslice, yslice] = f"{len(self._entries):^3}"
        # Assign the subspec to the entry
        entry.subspec = self._gs[xslice, yslice]
        self._entries.append(entry)

        # NOT USED YET
        entry.register_arguments(self._parser)

    def allocate_panels(self):
        """Allocate the panels for each entry in the dashboard."""
        for entry in self._entries:
            entry.allocate_panels(self._fig)

    def plot_panels(self, exclude_entries: Optional[list[str]] = None):
        """Plot every entry, skipping (and logging) any that fail.

        Args:
            exclude_entries: Entry names to skip.
        """
        for entry in self._entries:
            if exclude_entries and entry.name in exclude_entries:
                logger.info(f"... Skipping {entry.name}")
                continue

            logger.info(f"... Plotting {entry.name}")
            try:
                entry.plot_panel()
            except Exception as e:
                fail_msg = f"Failed plotting {entry.name}: {e}"
                logger.warning(fail_msg)
                logger.warning("Full traceback:\n" + traceback.format_exc())
                if not self._fail_silently:
                    raise Exception(fail_msg)

    def compute_data(
        self,
        recompute_entries: Optional[list[str]] = None,
        exclude_entries: Optional[list[str]] = None,
    ):
        """Compute data for every entry, caching to JSON.

        Args:
            recompute_entries: Entry names to force-recompute.
            exclude_entries: Entry names to skip.
        """
        if recompute_entries is None:
            recompute_entries = []
        if exclude_entries is None:
            exclude_entries = []

        for entry in self._entries:
            # Skip excluded entries
            if entry.name in exclude_entries:
                logger.info(
                    f"{entry.name} is in exclude_entries. Skipping it ..."
                )
                continue

            try:
                # Compute the data for the panel with optional recompute
                force_recompute = entry.name in recompute_entries
                entry.get_data(self._shared_ressources, force_recompute)
            except Exception as e:
                fail_msg = (
                    f"Failed computing {entry.name}: {e}, adding to "
                    "exclude list."
                )
                logger.warning(fail_msg)
                logger.warning("Full traceback:\n" + traceback.format_exc())
                # Add the entry to the exclude list to avoid plotting it
                exclude_entries.append(entry.name)
                if not self._fail_silently:
                    raise Exception(fail_msg)

    def render(
        self,
        *,
        suptitle: Optional[str] = None,
        recompute: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> "Dashboard":
        """Run the allocate -> compute -> plot lifecycle; optionally title it.

        A panel that fails to compute is added to a *shared* exclude list so it
        is also skipped at plot time (otherwise plotting an un-computed panel
        trips an assertion). Returns ``self`` for chaining.

        Args:
            suptitle: Figure-level title, if any.
            recompute: Entry names to force-recompute (ignore the JSON cache).
            exclude: Entry names to skip up front.
        """
        exclude_entries = list(exclude or [])
        self.allocate_panels()
        # compute_data appends any failures to this list in place; passing the
        # same object to plot_panels keeps the two stages in agreement.
        self.compute_data(
            recompute_entries=recompute, exclude_entries=exclude_entries
        )
        self.plot_panels(exclude_entries=exclude_entries)
        if suptitle:
            self._fig.suptitle(suptitle)
        return self

    def savefig(self, filepath):
        """Save the figure to ``filepath`` (layout is managed automatically).

        The figure uses matplotlib's constrained-layout engine, so spacing
        (including within panels' subgridspecs) is resolved at draw time --
        no explicit ``tight_layout`` call.
        """
        self._fig.savefig(filepath, bbox_inches="tight")
