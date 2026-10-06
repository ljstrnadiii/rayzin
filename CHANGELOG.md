# CHANGELOG


## v0.2.0 (2026-10-06)

### Bug Fixes

- Bound a manifest batch that a filter emptied
  ([`c9e635f`](https://github.com/ljstrnadiii/rayzin/commit/c9e635f08ee90f69287a5556f43f0d088580303f))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Documentation

- Parquet row groups stream to the GPU too
  ([`2d2b973`](https://github.com/ljstrnadiii/rayzin/commit/2d2b973bc6908e5700e005dd424c4e7e98b2e5b5))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Features

- Stream Parquet row groups to the GPU like COG blocks
  ([`6c66d1f`](https://github.com/ljstrnadiii/rayzin/commit/6c66d1f72b67cd1d25e2e28642e3c32adca51cf9))

One ranged request per run of adjacent column chunks, at offsets from a footer read once per file;
  decoded in the stored dtype into pinned buffers, with invalid or filtered rows as NaN, and scored
  by the torch backend's raw path. The block stream moves to readers/stream.py and takes any reader
  with fetch_encoded.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Performance Improvements

- Decode fetched row groups from a native sparse buffer
  ([`0d6c8fb`](https://github.com/ljstrnadiii/rayzin/commit/0d6c8fb3f78a519c1abfa0d300ae3229d770b971))

A Python file object made pyarrow take the GIL for every small read, so decode threads queued on it
  with the CPU idle.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Decode row groups on the reader's threads, not Arrow's pool
  ([`07e1642`](https://github.com/ljstrnadiii/rayzin/commit/07e164263511b0147ccb25ee3571bd968d1734d2))

Ray sets OMP_NUM_THREADS=1 in a one-CPU actor, which sizes Arrow's CPU pool to one thread, so eight
  decode threads queued on it: 194 MB/s vs 1,391 MB/s locally.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Read single row groups with ranged requests too
  ([`be0f245`](https://github.com/ljstrnadiii/rayzin/commit/be0f2458159bbcf95e90a85cb943835a364f7885))

read_row_group and read_rows fetched through a buffered file object, many small sequential reads per
  row group. They now fetch the needed column chunks in one request per adjacent run, like the
  stream.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>


## v0.1.0 (2026-10-06)

### Bug Fixes

- One Parquet handle per read, so concurrent row-group reads cannot crash
  ([`123cae2`](https://github.com/ljstrnadiii/rayzin/commit/123cae28a555b401c3709a05fa86769a69012f3f))

Several threads seeking and reading one cached file object segfaulted pyarrow on the cluster. Each
  read now opens its own handle and reuses the file's footer.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Code Style

- Ruff format the COG manifest summary
  ([`d482e81`](https://github.com/ljstrnadiii/rayzin/commit/d482e819dac4dfdbe600fa8a4174679b5e6f89dc))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Continuous Integration

- Keep torch and faiss on the same MKL on Linux; type unit_rows
  ([`7b81719`](https://github.com/ljstrnadiii/rayzin/commit/7b817194d4b1a41cfe5fc31b756a2a9ca9295d95))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Run pixi 0.71.1, which reads the regenerated lockfile
  ([`f83eefc`](https://github.com/ljstrnadiii/rayzin/commit/f83eefc9a32339b92b17777656fcfaab2ff39b1d))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Type-check the same on Linux and macOS
  ([`e782bf7`](https://github.com/ljstrnadiii/rayzin/commit/e782bf778f25f2c928cef7cae3474b8d9770a1a1))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Features

- Cogsearcher, a manifest loaded once and searched many times
  ([`fc18939`](https://github.com/ljstrnadiii/rayzin/commit/fc1893980c7fdc82cdafa85e58664fbd33433fa3))

knn_cog_search starts actors, reads the manifest and builds a heap on every call. CogSearcher keeps
  them: each actor holds its share of the manifest with a warm reader, backend and block stream, and
  one heap actor merges results across searches until close().

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Exact KNN over COG blocks with incremental manifests, AOI footprints and GPU FAISS
  ([`73f64c7`](https://github.com/ljstrnadiii/rayzin/commit/73f64c78c89ac5e615c95d82039147506a323a78))

- CogVectorReader fetches internal blocks with async-tiff and decodes them itself (zstd, deflate,
  none; float16/32/64), so float16 embedding COGs read without GDAL. - build_manifest_from_cogs
  writes one row per block (slice, count, centroid, radius, lon/lat footprint, epsg, metadata) and
  only scans URLs the manifest does not hold yet. - aoi filters COG manifests on stored footprints
  without opening files. - BlockSearcher prefetches surviving blocks; knn_cog_search takes
  num_gpus_per_actor and FAISS GPU reuses one StandardGpuResources per process. - PyPI extras: cog,
  cpu (faiss-cpu), gpu (faiss-gpu).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Exact KNN over vectors stored as Parquet rows, e.g. detections and their embeddings
  ([`522116f`](https://github.com/ljstrnadiii/rayzin/commit/522116fa39423b2a1f924a1b667562c1aef48324))

ParquetVectorReader reads a fixed-size list column a row group at a time through obstore, with an
  optional per-row filter (e.g. score); build_manifest_from_parquet summarises each row group with
  its bounding box, incrementally; knn_parquet_search searches them. The warm searcher, now
  KnnSearcher, takes either kind of manifest.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Manifests read and write through an optional pyarrow filesystem
  ([`ead7224`](https://github.com/ljstrnadiii/rayzin/commit/ead72245f11861ce59cfda48488b9c0195259906))

pyarrow's own GCS filesystem calls buckets.get to test a path, which object-scoped service accounts
  lack; a caller can hand in any pyarrow.fs.FileSystem instead.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Optional centroid pruning, and an explicit even batch size
  ([`d1b7f24`](https://github.com/ljstrnadiii/rayzin/commit/d1b7f2457c37baf7e1324079d440c0968648a1c4))

prune=False reads the manifest without centroids and radii, skips bounds and searches every block
  that survives pushdown: same exact top-k, for blocks too large or too many for bounds to pay off.
  Manifest rows are parsed column-wise instead of through Python lists. Ray needs a batch size for
  GPU map_batches, so the even split now sets it explicitly.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Performance Improvements

- Decode fetched COG blocks from async-tiff's buffer without copying it
  ([`031e43f`](https://github.com/ljstrnadiii/rayzin/commit/031e43f76232f49ef203097584f7a655472e61bd))

Copying each block into bytes held the GIL for tens of milliseconds and stalled the thread feeding
  the GPU: one node streamed 9.7 v2 blocks/s with the copy and 17.1 without.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Keep the torch top-k on the device and report search stats
  ([`828b101`](https://github.com/ljstrnadiii/rayzin/commit/828b101767457d2bfda2ac985411c937ec277b05))

TorchResultHeap merges a block's (nq, k) top-k with one topk instead of a Python heap push per
  candidate, and drains once per batch; the global heap merges whole result sets the same way. knn
  searches return blocks after pushdown, blocks and vectors searched in SearchResults.stats.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Queue a few even batches per actor so the pool grows to all of them
  ([`74adeb2`](https://github.com/ljstrnadiii/rayzin/commit/74adeb25bf70f8e27aefc426393b324df62d49e0))

One batch per actor left Ray Data's pool at a few actors on full scans, two to three times slower;
  several even batches each keep the pool growing and the tail short.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Split the blocks that survive pushdown evenly across search actors
  ([`3d589b8`](https://github.com/ljstrnadiii/rayzin/commit/3d589b872a89b4d082debb73f60d8a6de89cc61b))

With a fixed batch size, a pushdown leaving 620 blocks made three batches, so three of eight GPU
  actors did the work. Without a batch_size, the bounded manifest is now split into one share per
  actor.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Start search actors before reading the manifest
  ([`f2563c4`](https://github.com/ljstrnadiii/rayzin/commit/f2563c4971122fe6de76a9f7cca0116f2e78f5d5))

Each search now starts its GPU actors and heap actor first, so process start, torch import and CUDA
  set-up overlap with Ray Data reading, filtering and bounding the manifest. The surviving blocks
  are then split at exact row indices, one share per actor, and every actor is killed when the call
  returns: nothing persists between searches. Actors are capped to what the cluster can place,
  leaving a CPU for the manifest read.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Stream raw COG blocks to a torch backend that scores on the GPU
  ([`dd7b1d0`](https://github.com/ljstrnadiii/rayzin/commit/dd7b1d0402fd7e366276dc31eff030b1e7c62ac1))

RawBlockStream keeps blocks fetching and decompressing straight into pinned buffers while earlier
  ones are scored; TorchSearchBackend copies them to the GPU in their stored dtype and masks,
  normalizes, scores and takes the top k there. Euclidean lower bounds no longer build an (rows,
  queries, d) array, so large query batches fit.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Refactoring

- Simplify the COG stream, searcher loop and torch backend; cover them in CI
  ([`cff7908`](https://github.com/ljstrnadiii/rayzin/commit/cff79086fac08f3b12f72fb6e496f553a2929910))

- one compression dispatch (decode_block uses decode_block_into) and CogLayout.block_nbytes;
  RawBlockStream fetches through a public CogVectorReader.fetch_compressed - the searcher picks
  candidates in one generator for both read paths, and the stream releases its own blocks; the dead
  host-side search_raw is gone - queries are normalised in one place, unit_rows - CPU torch in the
  dev environment, plus tests for the block stream and the torch heap - README covers the COG and
  Parquet paths, the extras and KnnSearcher

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>


## v0.0.1 (2026-04-18)

### Bug Fixes

- Simplify __init__.py
  ([`802be91`](https://github.com/ljstrnadiii/rayzin/commit/802be91a730fa361cabaa5ac57b15a2bce38a955))
