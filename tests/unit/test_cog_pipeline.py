import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import shapely  # type: ignore[import-untyped]
from ray.data.expressions import col

from rayzin.pipeline import build_manifest_from_cogs, knn_cog_search
from rayzin.readers.cog_reader import CogVectorReader

BANDS, SIDE, BLOCK = 8, 32, 16
PLANTED = (20, 5)
GDAL = """
import json, sys
import numpy as np
import rasterio
from affine import Affine

for spec in json.loads(sys.argv[1]):
    values = np.load(spec["values"])
    with rasterio.open(
        spec["path"], "w", driver="GTiff", width=values.shape[2], height=values.shape[1],
        count=values.shape[0], dtype=spec["dtype"], crs="EPSG:32633",
        transform=Affine(28.8, 0, 500_000, 0, -28.8, 5_800_000), nodata=float("nan"), tiled=True,
        blockxsize=16, blockysize=16, interleave="pixel", compress=spec["compress"],
    ) as destination:
        destination.write(values.astype(spec["dtype"]))
    with rasterio.open(spec["path"]) as raster:
        np.save(spec["path"] + ".npy", raster.read())
"""


def write_cogs(tmp_path: Path, rasters: dict[str, tuple[np.ndarray, str, str]]) -> dict[str, str]:
    specs = []
    for name, (values, dtype, compress) in rasters.items():
        np.save(tmp_path / f"{name}.values.npy", values)
        specs.append(
            {
                "path": str(tmp_path / f"{name}.tif"),
                "values": str(tmp_path / f"{name}.values.npy"),
                "dtype": dtype,
                "compress": compress,
            }
        )
    subprocess.run([sys.executable, "-c", GDAL, json.dumps(specs)], check=True)
    return {name: str(tmp_path / f"{name}.tif") for name in rasters}


@pytest.fixture
def cogs(tmp_path: Path) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(0)
    target = rng.normal(size=BANDS).astype(np.float32)
    rasters = {}
    for year in (2024, 2025, 2026):
        values = rng.normal(size=(BANDS, SIDE, SIDE)).astype(np.float32)
        values[:, 0, 0] = np.nan
        if year == 2026:
            values[:, PLANTED[0], PLANTED[1]] = target
        rasters[str(year)] = (values, "float16", "zstd")
    return list(write_cogs(tmp_path, rasters).values()), target


@pytest.fixture
def manifest(tmp_path: Path, cogs: tuple[list[str], np.ndarray], ray_session: None) -> str:
    urls, _ = cogs
    path = str(tmp_path / "manifest")
    build_manifest_from_cogs(urls, path, metadata={"year": [2024, 2025, 2026]})
    return path


def test_the_reader_decodes_blocks_exactly_as_gdal_reads_them(tmp_path: Path) -> None:
    values = np.random.default_rng(1).normal(size=(BANDS, SIDE, SIDE)).astype(np.float32)
    kinds = [("float16", "zstd"), ("float32", "deflate"), ("float32", "none")]
    urls = write_cogs(tmp_path, {f"{d}-{c}": (values, d, c) for d, c in kinds})

    for url in urls.values():
        expected = np.load(url + ".npy")[:, 16:32, 0:16].reshape(BANDS, -1).T
        reader = CogVectorReader(normalize=False)
        [block] = reader.run(reader.fetch_blocks([(url, 16, 0)]))

        np.testing.assert_array_equal(block.vectors, expected.astype(np.float32))


def test_a_manifest_has_one_row_per_block_with_its_footprint_and_metadata(
    manifest: str, cogs: tuple[list[str], np.ndarray]
) -> None:
    urls, _ = cogs

    table = pq.read_table(manifest)

    assert table.num_rows == 3 * (SIDE // BLOCK) ** 2
    assert sorted(set(table.column("url").to_pylist())) == sorted(urls)
    assert sorted(table.column("count").to_pylist())[:3] == [BLOCK * BLOCK - 1] * 3
    footprint = shapely.from_wkb(table.column("geometry")[0].as_py())
    assert footprint.bounds[0] == pytest.approx(15.0, abs=0.5)
    assert set(table.column("year").to_pylist()) == {2024, 2025, 2026}


def test_a_rebuild_only_scans_files_the_manifest_does_not_hold(
    tmp_path: Path, cogs: tuple[list[str], np.ndarray], ray_session: None
) -> None:
    urls, _ = cogs
    path = str(tmp_path / "growing")

    first = build_manifest_from_cogs(urls[:2], path)
    second = build_manifest_from_cogs(urls, path)
    third = build_manifest_from_cogs(urls, path)

    assert (first, second, third) == (2, 1, 0)
    assert pq.read_table(path).num_rows == 3 * (SIDE // BLOCK) ** 2


def test_knn_over_cogs_finds_the_planted_vector_and_never_returns_nodata(
    manifest: str, cogs: tuple[list[str], np.ndarray]
) -> None:
    urls, target = cogs

    results = knn_cog_search(manifest, target[None, :], k=5, batch_size=4, actor_pool_size=1)

    top = results.chunks[0]
    assert top["url"] == urls[2]
    assert [(part["start"], part["stop"]) for part in top["slice"]] == [(16, 32), (0, 16)]
    assert results.offsets[0] == (PLANTED[0] - 16) * BLOCK + PLANTED[1]
    assert results.distances[0] == pytest.approx(0.0, abs=1e-5)
    assert all(distance < 4.0 for distance in results.distances)


def test_a_metadata_filter_prunes_files_before_they_are_read(
    manifest: str, cogs: tuple[list[str], np.ndarray]
) -> None:
    urls, target = cogs

    results = knn_cog_search(manifest, target[None, :], k=3, filter_expr=col("year") < 2026)

    assert urls[2] not in {chunk["url"] for chunk in results.chunks}
    assert results.stats["blocks_after_pushdown"] == 2 * (SIDE // BLOCK) ** 2


def test_an_aoi_keeps_only_blocks_whose_footprint_it_touches(
    manifest: str, cogs: tuple[list[str], np.ndarray]
) -> None:
    _, target = cogs
    table = pq.read_table(manifest)
    first_block = shapely.from_wkb(table.column("geometry")[0].as_py()).buffer(-1e-4)

    results = knn_cog_search(manifest, target[None, :], k=3, aoi=first_block)

    assert {
        tuple((part["start"], part["stop"]) for part in chunk["slice"]) for chunk in results.chunks
    } == {((0, 16), (0, 16))}


def test_the_torch_backend_streams_raw_blocks_to_the_same_neighbours_as_numpy(
    manifest: str, cogs: tuple[list[str], np.ndarray]
) -> None:
    pytest.importorskip("torch")
    from rayzin.enums import SearchBackendType

    urls, target = cogs
    queries = np.stack([target, np.random.default_rng(3).normal(size=BANDS).astype(np.float32)])

    streamed = knn_cog_search(
        manifest, queries, k=5, backend=SearchBackendType.TORCH, batch_size=16, prefetch=3
    )
    read = knn_cog_search(manifest, queries, k=5, batch_size=16)

    assert streamed.chunks[0]["url"] == urls[2]
    assert streamed.offsets[0] == (PLANTED[0] - 16) * BLOCK + PLANTED[1]
    assert sorted(zip(streamed.query_ids, streamed.offsets)) == sorted(
        zip(read.query_ids, read.offsets)
    )
    np.testing.assert_allclose(
        sorted(streamed.distances), sorted(read.distances), rtol=1e-4, atol=1e-5
    )


def test_without_pruning_the_search_skips_bounds_and_returns_the_same_neighbours(
    manifest: str, cogs: tuple[list[str], np.ndarray]
) -> None:
    _, target = cogs
    queries = np.stack([target, np.random.default_rng(5).normal(size=BANDS).astype(np.float32)])

    pruned = knn_cog_search(manifest, queries, k=5)
    exhaustive = knn_cog_search(manifest, queries, k=5, prune=False)

    assert sorted(zip(exhaustive.query_ids, exhaustive.offsets)) == sorted(
        zip(pruned.query_ids, pruned.offsets)
    )
    assert exhaustive.stats["blocks_searched"] == 3 * (SIDE // BLOCK) ** 2


def test_a_searcher_answers_repeated_and_filtered_searches_like_a_one_shot_search(
    manifest: str, cogs: tuple[list[str], np.ndarray]
) -> None:
    import pyarrow.compute as pc  # type: ignore[import-untyped]

    from rayzin.enums import SearchBackendType
    from rayzin.searcher import CogSearcher

    urls, target = cogs
    queries = np.stack([target, np.random.default_rng(7).normal(size=BANDS).astype(np.float32)])
    one_shot = knn_cog_search(manifest, queries, k=5)
    searcher = CogSearcher(manifest, actors=1, backend=SearchBackendType.NUMPY)
    try:
        first = searcher.search(queries, k=5)
        again = searcher.search(queries, k=5)
        filtered = searcher.search(queries, k=5, filter_expr=pc.field("year") < 2026)
    finally:
        searcher.close()

    expected = sorted(zip(one_shot.query_ids, one_shot.offsets))
    assert sorted(zip(first.query_ids, first.offsets)) == expected
    assert sorted(zip(again.query_ids, again.offsets)) == expected
    assert urls[2] not in {chunk["url"] for chunk in filtered.chunks}
    assert filtered.stats["blocks_after_pushdown"] == 2 * (SIDE // BLOCK) ** 2
