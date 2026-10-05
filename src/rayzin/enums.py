from enum import StrEnum


class MetricType(StrEnum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


class ReaderType(StrEnum):
    ZARR = "zarr"
    COG = "cog"
    PARQUET = "parquet"


class SearchBackendType(StrEnum):
    NUMPY = "numpy"
    FAISS_CPU = "faiss_cpu"
    FAISS_GPU = "faiss_gpu"
    TORCH = "torch"
