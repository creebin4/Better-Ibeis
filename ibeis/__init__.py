"""
IBEIS: minimal package init for ingest+match tests
"""

__version__ = '2.4.0'

# Ensure OpenCV is available (headless preferred)
try:
    import cv2  # NOQA
except ImportError as ex:
    import ubelt as ub
    msg = ub.paragraph(
        '''
        The ibeis module failed to import the cv2 module.
        Please install either:
          pip install opencv-python-headless
        or
          pip install opencv-python

        orig_ex={!r}
        '''
    ).format(ex)
    raise ImportError(msg)

# Core utilities (keep minimal footprint)
import utool as ut  # NOQA

# Export only the minimal public surface needed by tests
from ibeis.main_module import opendb  # NOQA
from ibeis.control.IBEISControl import IBEISController  # NOQA

# Optional: expose key HOTS classes (not required by the tests)
try:  # NOQA
    from ibeis.algo.hots.query_request import QueryRequest  # NOQA
    from ibeis.algo.hots.chip_match import ChipMatch  # NOQA
except Exception:
    # Keep import failures here non-fatal; tests only require opendb
    pass
