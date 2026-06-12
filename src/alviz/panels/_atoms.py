"""Shared helpers for reading values off ASE ``Atoms`` artifacts.

ASE's extxyz reader moves reserved properties (``energy``, ``forces``,
``stress``, ...) into a ``SinglePointCalculator``, so a label written as
``info["energy"]`` round-trips into ``atoms.calc.results["energy"]`` rather
than staying in ``info``. These resolvers look in the natural container
*and* in the calculator results, so any naming convention works.
"""

import numpy as np


def calc_results(atoms):
    """Return the attached calculator's results dict (empty if none)."""
    calc = getattr(atoms, "calc", None)
    return getattr(calc, "results", {}) if calc is not None else {}


def resolve_scalar(atoms, key):
    """Resolve a per-config scalar from ``info`` or calculator results."""
    if key in atoms.info:
        return float(atoms.info[key])
    results = calc_results(atoms)
    if key in results:
        return float(results[key])
    raise KeyError(
        f"scalar key {key!r} not found in atoms.info or calculator results"
    )


def resolve_array(atoms, key):
    """Resolve a per-atom array from ``arrays`` or calculator results."""
    if key in atoms.arrays:
        return np.asarray(atoms.arrays[key])
    results = calc_results(atoms)
    if key in results:
        return np.asarray(results[key])
    raise KeyError(
        f"array key {key!r} not found in atoms.arrays or calculator results"
    )


def as_paths(config_files):
    """Normalize a path or iterable of paths to a list."""
    if isinstance(config_files, (str, bytes)) or hasattr(
        config_files, "__fspath__"
    ):
        return [config_files]
    return list(config_files)
