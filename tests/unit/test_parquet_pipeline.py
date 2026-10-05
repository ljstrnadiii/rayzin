from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import shapely  # type: ignore[import-untyped]
from ray.data.expressions import col

from rayzin.manifest.parquet import geometry_bounds
from rayzin.pipeline import build_manifest_from_parquet, knn_parquet_search

DIM, ROWS, GROUPS = 8, 8, 2
PLANTED = (1, 3)


def write_boxes(path: Path, vectors: np.ndarray, scores: np.ndarray, x0: float) -> str:
    boxes = [
        shapely.box(x0 + i * 0.001, 52.0, x0 + i * 0.001 + 0.0005, 52.0005)
        for i in range(len(vectors))
    ]
    table = pa.table(
        {
            "geometry": pa.array(shapely.to_wkb(boxes), pa.binary()),
            "score": pa.array(scores, pa.float64()),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.astype(np.float16).ravel()), DIM
            ),
        }
    )
    pq.write_table(table, path, row_group_size=ROWS)
    return str(path)


@pytest.fixture
def boxes(tmp_path: Path) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(0)
    target = rng.normal(size=DIM).astype(np.float32)
    urls = []
    for index, year in enumerate((2024, 2025, 2026)):
        vectors = rng.normal(size=(ROWS * GROUPS, DIM)).astype(np.float32)
        scores = np.full(ROWS * GROUPS, 0.9)
        if year == 2026:
            vectors[PLANTED[0] * ROWS + PLANTED[1]] = target
            scores[PLANTED[0] * ROWS + PLANTED[1]] = 0.1
        urls.append(write_boxes(tmp_path / f"{year}.parquet", vectors, scores, 4.0 + index))
    return urls, target


@pytest.fixture
def manifest(tmp_path: Path, boxes: tuple[list[str], np.ndarray], ray_session: None) -> str:
    urls, _ = boxes
    path = str(tmp_path / "manifest")
    build_manifest_from_parquet(urls, path, metadata={"year": [2024, 2025, 2026]})
    return path


def test_a_manifest_has_one_row_per_row_group_with_its_bounding_box(
    manifest: str, boxes: tuple[list[str], np.ndarray]
) -> None:
    urls, _ = boxes

    table = pq.read_table(manifest)

    assert table.num_rows == 3 * GROUPS
    assert set(table.column("count").to_pylist()) == {ROWS}
    first = shapely.from_wkb(table.column("geometry")[0].as_py())
    assert first.bounds[1] == pytest.approx(52.0)
    assert sorted(set(table.column("url").to_pylist())) == sorted(urls)


def test_a_rebuild_only_reads_files_the_manifest_lacks(
    tmp_path: Path, boxes: tuple[list[str], np.ndarray], ray_session: None
) -> None:
    urls, _ = boxes
    path = str(tmp_path / "growing")

    assert build_manifest_from_parquet(urls[:2], path) == 2
    assert build_manifest_from_parquet(urls, path) == 1
    assert build_manifest_from_parquet(urls, path) == 0


def test_knn_over_parquet_finds_the_planted_vector_at_its_row(
    manifest: str, boxes: tuple[list[str], np.ndarray]
) -> None:
    urls, target = boxes

    results = knn_parquet_search(manifest, target[None, :], k=3, actor_pool_size=1)

    top = results.chunks[0]
    assert top["url"] == urls[2]
    assert [(part["dim"], part["start"]) for part in top["slice"]] == [("row_group", PLANTED[0])]
    assert results.offsets[0] == PLANTED[1]
    assert results.distances[0] == pytest.approx(0.0, abs=1e-5)


def test_a_row_filter_drops_single_rows_and_a_metadata_filter_whole_files(
    manifest: str, boxes: tuple[list[str], np.ndarray]
) -> None:
    urls, target = boxes

    confident = knn_parquet_search(
        manifest,
        target[None, :],
        k=3,
        row_filter=pc.field("score") > 0.5,
        row_filter_columns=("score",),
    )
    older = knn_parquet_search(manifest, target[None, :], k=3, filter_expr=col("year") < 2026)

    assert (confident.chunks[0]["url"], confident.offsets[0]) != (urls[2], PLANTED[1])
    assert all(distance > 1e-3 for distance in confident.distances)
    assert urls[2] not in {chunk["url"] for chunk in older.chunks}


def test_bounds_come_from_native_geoarrow_polygons_as_well_as_wkb() -> None:
    point = pa.struct([("x", pa.float64()), ("y", pa.float64())])
    rings = pa.array(
        [[[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 5.0}, {"x": 1.0, "y": 2.0}]]],
        pa.list_(pa.list_(point)),
    )

    assert geometry_bounds(rings) == (1.0, 2.0, 3.0, 5.0)


def test_a_searcher_over_parquet_answers_like_a_one_shot_search(
    manifest: str, boxes: tuple[list[str], np.ndarray]
) -> None:
    from rayzin.enums import ReaderType, SearchBackendType
    from rayzin.searcher import KnnSearcher

    _, target = boxes
    one_shot = knn_parquet_search(manifest, target[None, :], k=4)
    searcher = KnnSearcher(
        manifest, actors=1, backend=SearchBackendType.NUMPY, reader=ReaderType.PARQUET
    )
    try:
        warm = searcher.search(target[None, :], k=4)
    finally:
        searcher.close()

    assert sorted(zip(warm.offsets, warm.distances)) == pytest.approx(
        sorted(zip(one_shot.offsets, one_shot.distances))
    )


def test_reading_many_row_groups_of_one_file_at_once_is_safe(
    tmp_path: Path, boxes: tuple[list[str], np.ndarray]
) -> None:
    from rayzin.readers.parquet_reader import ParquetVectorReader, row_group_slice

    vectors = np.random.default_rng(9).normal(size=(ROWS * 64, DIM)).astype(np.float32)
    url = write_boxes(tmp_path / "many.parquet", vectors, np.full(len(vectors), 0.9), 4.0)
    reader = ParquetVectorReader(normalize=False, threads=16)
    chunks = [
        {
            "url": url,
            "slice": row_group_slice(group),
            "count": ROWS,
            "centroid": None,
            "radius": 0.0,
        }
        for group in range(64)
    ] * 4

    read = reader.read_many(chunks)  # type: ignore[arg-type]

    np.testing.assert_allclose(
        np.concatenate([vectors for vectors, _ in read[:64]]),
        vectors.astype(np.float16).astype(np.float32),
    )
