"""Machine Learning module for NeuroImaging in python."""
import gzip

try:
    from nilearn._version import __version__  # noqa: F401
except ImportError:
    __version__ = "0+unknown"

# Monkey-patch gzip to have faster reads on large gzip files
if hasattr(gzip.GzipFile, "max_read_chunk"):
    gzip.GzipFile.max_read_chunk = 100 * 1024 * 1024  # 100Mb

# Boolean controlling the default globbing technique when using check_niimg
# and the os.path.expanduser usage in CacheMixin.
# Default value it True, set it to False to completely deactivate this
# behavior.
EXPAND_PATH_WILDCARDS = True

# list all submodules available in nilearn and version
from nilearn import datasets, decoding, decomposition, connectome, experimental
from nilearn import image, maskers, masking, interfaces, mass_univariate
from nilearn import plotting, regions, signal, surface

# Import nilearnx extensions
from .extensions import *


__all__ = [
    "datasets",
    "decoding",
    "decomposition",
    "connectome",
    "experimental",
    "image",
    "maskers",
    "masking",
    "interfaces",
    "mass_univariate",
    "plotting",
    "regions",
    "signal",
    "surface",
    "__version__",
    "extensions",
    "dataframe_functions"
]
