__version__ = "0.0.1"

from rayzin.enums import MetricType, SearchBackendType
from rayzin.manifest import selector_from_aoi
from rayzin.pipeline import (
    build_manifest,
    build_manifest_from_cogs,
    build_manifest_from_zarr,
    knn_cog_search,
    knn_zarr_search,
)
from rayzin.types import SearchResults

__all__ = [
    "__version__",
    # pipeline
    "knn_zarr_search",
    "knn_cog_search",
    # pipeline — manifest build
    "build_manifest",
    "build_manifest_from_zarr",
    "build_manifest_from_cogs",
    "selector_from_aoi",
    # pipeline API types
    "MetricType",
    "SearchBackendType",
    "SearchResults",
]
