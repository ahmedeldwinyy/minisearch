# MiniSearch Recommender

**Data note:** SciFact has documents and relevance labels, but no user interaction
data. The recommender's users and ratings will be synthetic; their results show
that the implementation and math work, not real-world recommendation quality.

## D2: Cosine Similarity Speed

Measured on the full SciFact index (5,183 documents, 35,941 terms), using document
`10009203` as the query and the top 10 recommendations. Each implementation ran
50 times; result document IDs matched, with scores equal within floating-point
tolerance. The measurement was taken on the local Apple Silicon development
machine and is a comparison of these two implementations, not a general hardware
benchmark.

| Implementation | Average time per lookup |
| --- | ---: |
| SciPy sparse vectorized dot product | 3.176 ms |
| Plain-Python CSR loop | 152.893 ms |

The vectorized implementation was 48.1x faster in this run.

## Synthetic Recommendation Evaluation

Further recommender results will be added after synthetic interactions,
matrix-factorization training, and cold-start evaluation are implemented.