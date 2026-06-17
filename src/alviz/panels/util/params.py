"""Render a free-form parameter dict as text on the canvas.

* ``ParamsPanel``     -- abstract: renders any dict (free-form ``JsonDict``,
                         no typed contract -- the content is arbitrary).
* ``JsonParamsPanel`` -- reads the dict from a JSON file.

Useful as a metadata panel (training params, run config, ...).
"""

from abc import abstractmethod

from alviz.dashboard import Entry
from alviz.mpl_utils import JsonDict, plot_dict_on_canvas


class ParamsPanel(Entry):
    """Display a dictionary as indented text.

    The data is a free-form ``JsonDict`` (``data_class`` stays the default),
    since the content is arbitrary parameters rather than a fixed shape.

    Args:
        title: Panel title.
        fontsize: Text size.
        width: Wrap width for the rendered text.
    """

    name = "params"

    def __init__(
        self,
        *args,
        title: str = "",
        fontsize: int = 10,
        width: int = 30,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._title = title
        self._fontsize = fontsize
        self._width = width

    def allocate_panels(self, fig):
        self._fig = fig
        self.gs = self.subspec.subgridspec(1, 1)
        self.ax = fig.add_subplot(self.gs[0, 0])

    @abstractmethod
    def compute_data(self, shared_ressources: dict) -> JsonDict:
        """Return the parameter dict (a JsonDict)."""
        raise NotImplementedError

    def plot_data(self):
        plot_dict_on_canvas(
            data=dict(self.data),
            title=self._title,
            fontsize=self._fontsize,
            width=self._width,
            ax=self.ax,
        )


class JsonParamsPanel(ParamsPanel):
    """Read the parameter dict from a JSON file.

    Args:
        input_file: Path to a JSON file.
    """

    def __init__(self, input_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_file = input_file

    def compute_data(self, shared_ressources: dict) -> JsonDict:
        return JsonDict.from_json(self._input_file)
