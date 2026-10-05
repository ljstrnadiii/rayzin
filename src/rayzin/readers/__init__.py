from typing import TYPE_CHECKING, Any

from rayzin.readers.protocol import VectorReader
from rayzin.readers.zarr_reader import ZarrVectorReader

if TYPE_CHECKING:
    from rayzin.enums import ReaderType


def make_reader(reader_type: "ReaderType", **kwargs: Any) -> VectorReader:
    from rayzin.enums import ReaderType

    if reader_type == ReaderType.ZARR:
        return ZarrVectorReader(**kwargs)
    if reader_type == ReaderType.COG:
        from rayzin.readers.cog_reader import CogVectorReader

        return CogVectorReader(**kwargs)
    if reader_type == ReaderType.PARQUET:
        from rayzin.readers.parquet_reader import ParquetVectorReader

        return ParquetVectorReader(**kwargs)
    raise ValueError(reader_type)


__all__ = ["VectorReader", "ZarrVectorReader", "make_reader"]
