"""Training panels: model accuracy vs. reference (parity, error)."""

from alviz.panels.training.energy_error import (
    EnergyErrorData,
    EnergyErrorPanel,
    ExtxyzEnergyErrorPanel,
)
from alviz.panels.training.parity import (
    ExtxyzParityPanel,
    ParityData,
    ParityPanel,
    parity_scatter,
)

__all__ = [
    "ParityPanel",
    "ExtxyzParityPanel",
    "ParityData",
    "parity_scatter",
    "EnergyErrorPanel",
    "ExtxyzEnergyErrorPanel",
    "EnergyErrorData",
]
