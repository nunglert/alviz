"""Concrete dashboard panels.

Each panel comes in two layers:

* an **abstract** base that owns the rendering and the data contract (a
  ``PanelData`` subclass), leaving ``compute_data`` to subclasses;
* a concrete ``Extxyz*`` reader that produces that data from extxyz files
  (the common case).

Projects with other data sources subclass the abstract base and implement
``compute_data`` -- getting the plot + validated contract for free. Panels
may require the ``panels`` extra (``ase``).
"""

from alviz.panels.energy_error import (
    EnergyErrorData,
    EnergyErrorPanel,
    ExtxyzEnergyErrorPanel,
)
from alviz.panels.parity import (
    ExtxyzParityPanel,
    ParityData,
    ParityPanel,
    parity_scatter,
)
from alviz.panels.sampled_ev import (
    ExtxyzSampledEnergyVolume,
    SampledEnergyVolume,
    SampledEVData,
)
from alviz.panels.uncertainty import (
    ExtxyzSampleUncertainties,
    SampleUncertainties,
    UncertaintyData,
)

__all__ = [
    # parity
    "ParityPanel",
    "ExtxyzParityPanel",
    "ParityData",
    "parity_scatter",
    # sampled energy/force/volume
    "SampledEnergyVolume",
    "ExtxyzSampledEnergyVolume",
    "SampledEVData",
    # ensemble uncertainty
    "SampleUncertainties",
    "ExtxyzSampleUncertainties",
    "UncertaintyData",
    # energy error
    "EnergyErrorPanel",
    "ExtxyzEnergyErrorPanel",
    "EnergyErrorData",
]
