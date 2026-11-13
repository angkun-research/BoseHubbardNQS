"""Collision-free package view of the original 2D implementation.

Use this route for integrations. Legacy standalone scripts remain supported in
separate interpreters; mixing routes is rejected before model classes diverge.
"""
from pathlib import Path
import sys

_SOURCE = Path(__file__).resolve().parents[1] / '2D'
for _name, _module in tuple(sys.modules.items()):
    _file = getattr(_module, '__file__', None)
    if _file and '.' not in _name and Path(_file).resolve().parent == _SOURCE:
        raise ImportError('2D legacy modules already loaded; use a fresh interpreter for pythonTest.bosehubbard2d')
__path__ = [str(_SOURCE)]
