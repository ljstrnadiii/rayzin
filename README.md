# `rayzin`
[![CI](https://github.com/ljstrnadiii/rayzin/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ljstrnadiii/rayzin/actions/workflows/ci.yml)
[![Release](https://github.com/ljstrnadiii/rayzin/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/ljstrnadiii/rayzin/actions/workflows/release.yml)

`rayzin` is a Ray project for exact k-nearest-neighbor search over chunked embedding arrays stored
in Zarr.

It is designed for workflows where:
- Embeddings are stored as chunked arrays, typically with a trailing feature dimension.
- Search should prune whole chunks before loading vector payloads whenever possible.
- Relevant chunks are read once, searched in parallel with Ray, and merged into a global top-k.

The project uses a chunk-first execution model:
1. Build or read a manifest describing each chunk by slice, count, centroid, and radius.
2. Apply manifest-level pushdown with Ray Data using predicate filters and optional AOI filtering.
3. Compute query-to-chunk lower bounds in vectorized batches.
4. Search only the active query subset for each surviving chunk after pruning.
5. Merge local top-k results into a global heap actor and return flat result rows.

This architecture keeps chunk reads bounded, avoids rescanning the same data for each query, and
fits naturally into Ray Data execution.

## Features

- **Manifest-driven pruning:** Chunk summaries let the pipeline skip work before loading dense
  vectors, which is the main lever for scaling exact search.
- **Ray-native distributed execution:** Manifest filtering, chunk reads, and search fan out through
  Ray Data while final top-k merging stays centralized in a heap actor.
- **Pluggable search backends:** NumPy and FAISS backends share the same pipeline so pruning and
  result handling stay consistent across implementations.

## Current Scope

- Zarr-backed search path.
- Batched query input shaped `(1, d)` or `(nq, d)`.
- Euclidean pruning path.
- NumPy backend by default, with FAISS supported when installed separately.
- Early COG entry points are present, but the COG path is not the main supported flow yet.

## Installation

Install the base package from PyPI:

```bash
pip install rayzin
```

That gives you the core Zarr search path and the NumPy backend.

If you want the FAISS backend, install FAISS separately after installing `rayzin`:

```bash
pip install faiss-cpu
```

For FAISS GPU, use your platform's recommended FAISS installation method. In practice that is
often conda or Pixi on `linux-64`.

If you are working from a local checkout and want an editable install:

```bash
pip install -e .
```

## Development

- Install Pixi: `curl -fsSL https://pixi.sh/install.sh | sh`
- Install dependencies: `pixi install -e dev`
- Run lint: `pixi run -e dev ruff check .`
- Run format check: `pixi run -e dev ruff format --check .`
- Run type check: `pixi run -e dev mypy src/rayzin tests`
- Run tests: `pixi run -e dev pytest`

## Release

Releases are cut automatically from `main` after the `CI` workflow passes. Use Conventional
Commits for changes that should trigger a release, and semantic-release will:

- compute the next version
- update `pyproject.toml` and `src/rayzin/__init__.py`
- refresh `pixi.lock`
- create the version tag and GitHub release
- build and publish the package to PyPI via GitHub Actions trusted publishing

## Roadmap

1. Keep pruning metrics and search-time metrics aligned across all supported backends.
2. Generalize pruning beyond a single radius or tau so multiple pivots or metric-independent
   pruning are possible.
3. Simplify filtering to chunk existence, AOI intersection, and straightforward coordinate
   subsetting.
4. Add support for native dtypes when no scale or offset or other CF decoding is required.
5. Add support for pre-transforms such as dequantization before search.
6. Prefer `from __future__ import annotations` over quoted forward references.
7. Add logging and timing around pruning, loading, decoding, and per-block search.
8. Revisit float32 assumptions and document where they are required by FAISS.
9. Make radius-based manifest data optional if pruning mode does not require it.
10. Revisit rows-per-block tuning and execution sizing once the main search path stabilizes.
