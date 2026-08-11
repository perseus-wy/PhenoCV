# -*- coding: utf-8 -*-
"""Allow ``python -m phenocv`` as an alias for the ``phenocv`` console script."""

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
