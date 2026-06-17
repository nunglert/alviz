"""Concrete dashboard panels, grouped by theme.

Each panel comes in two layers:

* an **abstract** base that owns the rendering and the data contract (a
  ``PanelData`` subclass), leaving ``compute_data`` to subclasses;
* a concrete ``Extxyz*`` reader that produces that data from extxyz files
  (the common case).

Projects with other data sources (a model, a database, NS postprocessing)
subclass the abstract base and implement ``compute_data`` -- getting the plot
+ validated contract for free.

Panels are organized into theme subpackages (``alviz.panels.training``,
``alviz.panels.sampling``, ``alviz.panels.nested_sampling``, ...). Everything
is also re-exported here for convenience: ``from alviz.panels import
ParityPanel`` keeps working.
"""

from alviz.panels.nested_sampling import (
    ExtxyzSampleUncertainties,
    HeatmapData,
    HeatmapPanel,
    SampleUncertainties,
    UncertaintyData,
    heatmap_pcolormesh,
)
from alviz.panels.sampling import (
    ExtxyzSampledEnergyVolume,
    SampledEnergyVolume,
    SampledEVData,
)
from alviz.panels.training import (
    EnergyErrorData,
    EnergyErrorPanel,
    ExtxyzEnergyErrorPanel,
    ExtxyzParityPanel,
    ParityData,
    ParityPanel,
    parity_scatter,
)
from alviz.panels.util import JsonParamsPanel, ParamsPanel

__all__ = [
    # training
    "ParityPanel",
    "ExtxyzParityPanel",
    "ParityData",
    "parity_scatter",
    "EnergyErrorPanel",
    "ExtxyzEnergyErrorPanel",
    "EnergyErrorData",
    # sampling
    "SampledEnergyVolume",
    "ExtxyzSampledEnergyVolume",
    "SampledEVData",
    # nested_sampling
    "SampleUncertainties",
    "ExtxyzSampleUncertainties",
    "UncertaintyData",
    "HeatmapPanel",
    "HeatmapData",
    "heatmap_pcolormesh",
    # util
    "ParamsPanel",
    "JsonParamsPanel",
]
