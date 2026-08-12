# -*- coding: utf-8 -*-
"""PhenoCV — a composable, open-source computer-vision toolkit for plant phenotyping.

PhenoCV is **not** a single algorithm; it is a toolbox of independent,
pluggable modules that share one core (the trait-extractor registry + IO
helpers in :mod:`phenocv.core`). Each module is a self-contained capability
you can adopt à la carte:

* :mod:`phenocv.segmentation` — temporal canopy segmentation via SAM 2 video
  propagation (data-source-agnostic engine + pluggable adapters).
* :mod:`phenocv.phenotypes` — a 4-tier, fail-closed trait engine (2D shape →
  RGB vegetation indices → 3D height/volume → multispectral indices) that runs
  every registered extractor whose inputs you actually have.

Extending PhenoCV
-----------------
Add a new module (e.g. ``phenocv.counting``) as a sibling package, register its
compute tools with :func:`phenocv.core.registry.register`, and they become
visible to :func:`list_modules` / the CLI automatically. No core change needed.

Use :func:`list_modules` to discover what is installed in the current
environment.
"""

__version__ = "0.1.0"


def list_modules() -> list[str]:
    """Return the names of installed PhenoCV capability modules.

    A module is "installed" when its package imports without raising (some
    modules need optional deps such as ``torch`` for SAM 2).
    """
    known = ["segmentation", "phenotypes", "thermal"]
    installed = []
    import importlib

    for name in known:
        try:
            importlib.import_module("phenocv.%s" % name)
            installed.append(name)
        except Exception:
            # Optional dependency missing — module reported but not loadable.
            pass
    return installed
