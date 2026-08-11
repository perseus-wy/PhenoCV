# -*- coding: utf-8 -*-
"""Adapter base class.

An adapter translates a user's dataset layout into the
:class:`~phenocv.engine.PlantSequence` objects the core engine consumes
(sequences + anchor masks). Implement ``build_sequences`` for any dataset
format; the default :class:`~phenocv.adapters.csv_manifest.CsvManifestAdapter`
already covers the common case of a single manifest CSV/JSON.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..engine import PlantSequence


class BaseAdapter(ABC):
    """Translate a dataset into engine-ready sequences."""

    @abstractmethod
    def build_sequences(self, **kwargs) -> List[PlantSequence]:
        """Return sequences (with anchors) ready for the engine.

        Implementations should silently skip sequences with too few anchors,
        but must report skips (e.g. via a ``progress`` callback) rather than
        dropping plants without a trace.
        """
        raise NotImplementedError
