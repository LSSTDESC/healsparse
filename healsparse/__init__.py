try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("healsparse")
except PackageNotFoundError:
    # package is not installed
    pass

from .healSparseMap import HealSparseMap
from .healSparseCoverage import HealSparseCoverage
from .healSparseRandoms import make_uniform_randoms, make_uniform_randoms_fast
from .operations import sum_union, sum_intersection
from .operations import product_union, product_intersection
from .operations import or_union, or_intersection
from .operations import and_union, and_intersection
from .operations import xor_union, xor_intersection
from .operations import max_intersection, min_intersection, max_union, min_union
from .operations import ufunc_union, ufunc_intersection
from .operations import divide_intersection, floor_divide_intersection
from . import geom
from .geom import Box, Circle, Ellipse, Polygon, realize_geom
from .utils import WIDE_MASK
from .cat_healsparse_files import cat_healsparse_files
