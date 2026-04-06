# rayzin

`rayzin` is a Ray-based KNN search pipeline for chunked embedding arrays stored in Zarr.

The current structure is:

- Build a manifest over Zarr chunks in index-space, storing each chunk `slice`, `count`, `centroid`, and `radius`.
- Apply shared manifest pushdown once with Ray Data using predicate filters and optional AOI filtering.
- Compute vectorized lower bounds for batched queries shaped `(nq, d)`.
- Search only the active query subset for each chunk after pruning, while reading each chunk once.
- Merge local top-k results into a global heap actor so workers do not return and re-sort large intermediate result sets.
- Return flat search results tagged by `query_id`, `chunk`, `offset`, and `distance`.

Current scope:

- Zarr-backed search only
- Batched query input only: `(1, d)` or `(nq, d)`
- Euclidean pruning path
- NumPy and FAISS search backends

## Roadmap

1. Keep pruning metrics and search-time metrics aligned across all supported backends.
2. Generalize pruning beyond a single radius/tau so multiple pivots or metric-independent pruning are possible.
3. Simplify filtering to chunk existence, AOI intersection, and straightforward coordinate subsetting.
4. Add support for native dtypes when no scale/offset or other CF decoding is required.
5. Add support for pre-transforms such as dequantization before search.
6. Prefer `from __future__ import annotations` over quoted forward references.
7. Add logging and timing around pruning, loading, decoding, and per-block search.
8. Revisit float32 assumptions and document where they are required by FAISS.
9. Make radius-based manifest data optional if pruning mode does not require it.
10. Revisit rows-per-block tuning and execution sizing once the main search path stabilizes.
