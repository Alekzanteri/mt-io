# package file
from .geomag02 import geomag02, read_geomag02
from .geomag_collection import GeomagCollection

__all__ = [
    "geomag02",
    "read_geomag02",
    "GeomagCollection"
]