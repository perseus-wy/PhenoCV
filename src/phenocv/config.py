# -*- coding: utf-8 -*-
"""Load propagation configuration from YAML (base params + optional preset).

The YAML layout mirrors the reference config::

    propagation:
      roi_pad_ratio: 1.9
      ...
    presets:
      plant_phenotyping:
        ...
      rigid_object:
        ...

``load_config`` returns a flat dict of :class:`TemporalPropagationConfig`
fields, ready to pass to ``engine.run_sam2_video_temporal(config=...)``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def load_config(path: Optional[str] = None,
                preset: Optional[str] = None) -> Dict[str, Any]:
    """Load a YAML config and merge ``preset`` over the base ``propagation``.

    Parameters
    ----------
    path : YAML file path. When None, an empty base is used.
    preset : preset name under the ``presets`` block to overlay on ``propagation``.

    Returns
    -------
    Flat dict of config fields.
    """
    data: Dict[str, Any] = {}
    if path and os.path.exists(path):
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    base = dict(data.get("propagation", {}))
    presets = data.get("presets", {}) or {}
    if preset:
        if preset not in presets:
            raise KeyError(
                "Unknown preset %r (available: %s)" % (preset, list(presets)))
        base.update(presets[preset])
    return base
