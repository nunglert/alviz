"""Concrete dashboard panels.

Each panel reads *producer artifacts* (files with a documented schema) and
never imports a producer package's internals. Panels may require optional
dependencies (e.g. ``ase`` for reading extxyz); install the ``panels``
extra: ``pip install "alviz[panels]"``.
"""

from alviz.panels.parity import ParityPanel, parity_scatter

__all__ = ["ParityPanel", "parity_scatter"]
