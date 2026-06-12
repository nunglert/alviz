"""Matplotlib helpers and the JSON-backed data container for panels.

This module is intentionally domain-agnostic: it depends only on the
standard library, numpy and matplotlib. ``JsonDict`` is the contract every
dashboard panel uses to store and reload its computed data, so panel data is
always a plain JSON-serializable mapping on disk.
"""

import json
import re
import textwrap
from collections.abc import MutableMapping
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle

LETTERS = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)", "i)", "j)"]


def load_rc_params():
    """Apply a consistent default matplotlib style.

    For all available options, see:
    https://matplotlib.org/stable/users/explain/customizing.html
    """
    import matplotlib as mpl

    mpl.rcParams["figure.figsize"] = (8, 6)
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["axes.linewidth"] = 2
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["lines.linewidth"] = 2

    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.major.size"] = 5
    mpl.rcParams["xtick.minor.size"] = 5
    mpl.rcParams["ytick.major.size"] = 5
    mpl.rcParams["ytick.minor.size"] = 5
    mpl.rcParams["xtick.major.width"] = 1.5
    mpl.rcParams["xtick.minor.width"] = 1
    mpl.rcParams["ytick.major.width"] = 1.5
    mpl.rcParams["ytick.minor.width"] = 1


class JsonDict(MutableMapping):
    """A dict that enforces JSON-serializability and round-trips to disk.

    Panels compute their data into a ``JsonDict`` so the dashboard can cache
    and reload it without re-running the (potentially expensive) computation.
    """

    def __init__(self, *args, **kwargs):
        self._data = dict()
        self.update(dict(*args, **kwargs))

    def _ensure_serializable(self, key, value):
        try:
            json.dumps(value)
        except (TypeError, OverflowError) as e:
            raise ValueError(f"Entry {key!r} is not JSON serializable: {e}")

    def keys(self) -> list[str]:
        return list(self._data.keys())

    # Core dict interface
    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._ensure_serializable(key, value)
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"

    # JSON export
    def to_json(self, path, indent=2):
        with open(path, "w") as f:
            json.dump(self._data, f, indent=indent)

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"JSON root must be an object, got {type(data)}")
        return cls(data)


def lighten_color(color_name, factor=0.3):
    """Lighten a matplotlib color (e.g. 'red', 'black', 'grey').

    ``factor`` controls how much white is added (0 = original, 1 = white).
    """
    color = np.array(mcolors.to_rgb(color_name))
    white = np.array([1, 1, 1])
    return (1 - factor) * color + factor * white


def merge_default_args(new_args: dict, defaults: dict) -> dict:
    """Merge ``new_args`` over a copy of ``defaults`` (defaults preserved)."""
    merged = defaults.copy()
    if new_args:
        merged.update(new_args)
    return merged


def get_color_sequence(cmap, size, start=None, plot=False):
    """Return ``size`` colors sampled across a colormap."""
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if start is None:
        if isinstance(cmap, LinearSegmentedColormap):
            start = 0.2
        elif isinstance(cmap, ListedColormap):
            start = 0
    cs = cmap(np.linspace(start, 1.0, size))
    if plot:
        for idx, c in enumerate(cs):
            plt.plot(np.ones(3) * idx, c=c)
    return cs


def get_const_colormap(r, g, b):
    """A colormap that is a single constant color."""
    return LinearSegmentedColormap(
        "constant",
        {
            "red": [(0, r, r), (1, r, r)],
            "green": [(0, g, g), (1, g, g)],
            "blue": [(0, b, b), (1, b, b)],
        },
    )


def make_letter(
    fig,
    outer_gs,
    ypos=1.1,
    xshift=0.0,
    letter="a)",
    label="",
    fontweight="bold",
):
    """Add a panel caption letter (e.g. ``a)``) above a subplot spec."""
    caption_ax = fig.add_subplot(outer_gs)
    caption_ax.axis("off")
    caption_ax.text(
        0.5 + xshift,
        ypos,
        label,
        ha="center",
        va="center",
        transform=caption_ax.transAxes,
        fontsize=12,
        fontweight=fontweight,
    )
    caption_ax.text(
        0.0 + xshift,
        ypos,
        letter,
        ha="center",
        va="center",
        transform=caption_ax.transAxes,
        fontsize=12,
        fontweight=fontweight,
    )


def format_ax_in_gs(
    ax, sub_spec: gridspec.SubplotSpec, xlabel: str, ylabel: str
) -> None:
    """Drop inner labels/ticklabels based on the panel's grid position."""
    if sub_spec.is_first_col():
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    if sub_spec.is_last_row():
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])


def style_ticks(
    ax,
    remove_x=True,
    remove_y=True,
    remove_labels=True,
    direction="in",
    tick_length=3,
    tick_width=1,
):
    """Style axis ticks: optionally hide labels, point ticks inward."""
    if remove_labels:
        if remove_x:
            ax.tick_params(axis="x", labelbottom=False)
        if remove_y:
            ax.tick_params(axis="y", labelleft=False)

    ax.tick_params(
        axis="both",
        direction=direction,
        length=tick_length,
        width=tick_width,
    )


def latexify_spg(spg_string):
    """Render a space-group string with a LaTeX overbar for ``-N``."""
    match = re.search("-(.)", spg_string)
    if match:
        arg = match.groups()[0]
        spg_string = re.sub("-(.)", rf"\\bar{{{arg}}}", spg_string)
    return rf"${spg_string}$"


def draw_box(fig, element, zorder=0, padding=(1.0, 1, 1.0, 1.0)):
    """Draw a translucent bounding box around a grid element's axes."""
    hide_axes = True
    if isinstance(element, gridspec.GridSpec):
        axes = [
            fig.add_subplot(element[i])
            for i in range(element.nrows * element.ncols)
        ]
    elif isinstance(element, gridspec.SubplotSpec):
        element = gridspec.GridSpecFromSubplotSpec(1, 1, element)
        axes = [
            fig.add_subplot(element[i])
            for i in range(element.nrows * element.ncols)
        ]
    else:
        axes = element  # Assume it's a list of Axes
        hide_axes = False

    bbox = [
        ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted()
        )
        for ax in axes
    ]

    # Compute the enclosing rectangle's bounds
    x_min = min(b.x0 for b in bbox)
    x_max = max(b.x1 for b in bbox)
    y_min = min(b.y0 for b in bbox)
    y_max = max(b.y1 for b in bbox)
    x_min -= (padding[0] - 1.0) * x_min
    x_max += (padding[1] - 1.0) * x_max
    y_min -= (padding[2] - 1.0) * y_min
    y_max += (padding[3] - 1.0) * y_max

    # Add padding (e.g., 2% of the figure size)
    padding = 0.02
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    width = x_max - x_min
    height = y_max - y_min

    rect = Rectangle(
        (x_min, y_min),
        width,
        height,
        edgecolor="None",
        facecolor="grey",
        linewidth=1,
        zorder=zorder,
        alpha=0.1,
    )
    rect_border = Rectangle(
        (x_min, y_min),
        width,
        height,
        edgecolor="black",
        facecolor="none",
        linewidth=1,
        zorder=zorder,
    )
    fig.add_artist(rect)
    fig.add_artist(rect_border)

    if hide_axes:
        [ax.set_visible(False) for ax in axes]


def plot_dict_on_canvas(
    data: dict,
    title: str = "Dict Contents",
    fontsize: int = 10,
    ax: Optional[Axes] = None,
    width: Optional[int] = None,
):
    """Render a JSON-like dictionary as indented text on an axis.

    Args:
        data: Dictionary to display.
        title: Plot title.
        fontsize: Font size for text display.
        ax: Axis to draw on (a new one is created if None).
        width: Wrap text to the given width.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")
    ax.set_title(title, fontsize=fontsize + 2, loc="left")

    text = json.dumps(data, indent=2)
    if width is not None:
        text = "\n".join(
            textwrap.fill(line, width=width) for line in text.splitlines()
        )

    ax.text(
        0.01,
        0.99,
        text,
        va="top",
        ha="left",
        fontsize=fontsize,
        family="monospace",
    )
