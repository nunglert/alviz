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

from alviz.mpl_utils import JsonDict

logger = logging.getLogger(__name__)


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
        compute_data(self, shared_ressources): Compute and return a JsonDict.
        plot_data(self): Plot the computed data on the allocated subplots.
    """

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

        # Documented fallback: an unnamed entry takes its class name, so the
        # framework is usable without every subclass declaring ``name``.
        self.name = name if name is not None else type(self).__name__

    def get_data(self, shared_ressources, force_recompute) -> JsonDict:
        """Get or compute the data for this entry.

        Args:
            shared_ressources: Shared ressources among entries.
            force_recompute: Whether to force recomputation of data.

        Returns:
            JsonDict containing the computed data.
        """
        # File to store/load the data
        json_file = self.datapath / f"{self.name}.json"
        # Check for recomputation flag in the entry or globally
        force_recomp = force_recompute or self._force_recompute
        # Load from file if it exists and no recomputation is forced
        if os.path.exists(json_file) and not force_recomp:
            logger.info(f"... Loading from file: {self.name}")
            self.data = JsonDict.from_json(json_file)
        else:
            logger.info(f"... Computing: {self.name}")
            # Compute the data
            self.data = self.compute_data(shared_ressources)
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
        """Compute and return data as a JsonDict."""
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
    def data(self, val: JsonDict):
        assert isinstance(val, JsonDict), "Only support JsonDict currently!"
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

        self._fig = plt.figure(
            figsize=(width_per_panel * ncols, height_per_panel * nrows)
        )

        # Handles the layout of the figure
        self._gs = GridSpec(
            nrows,
            ncols,
            figure=self._fig,
            height_ratios=[1.0] * nrows,
        )

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

    def savefig(self, filepath):
        """Tight-layout and save the figure to ``filepath``."""
        self._fig.tight_layout()
        self._fig.savefig(filepath, bbox_inches="tight")
