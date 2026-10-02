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

SciFact contains documents but no user-rating data. All user IDs, topic preferences, and ratings below are synthetic, generated with fixed seeds from 5,183 documents; results validate the code and math, not real-world recommendation quality.

The matrix-factorization and baseline comparison uses the held-out ratings of 1,000 synthetic users.

MF settings were selected by validation RMSE on a 20% per-user split of outer training ratings only: n_factors=24, lr=0.015, reg=0.03, epochs=12 (validation RMSE 1.003 across 6,000 ratings). The held-out test split was not used for parameter selection.

### Matrix Factorization vs Baselines

| Strategy | Test RMSE | Precision@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| popularity | N/A | 0.010 | 0.017 |
| random | N/A | 0.001 | 0.002 |
| matrix_factorization | 1.004 | 0.002 | 0.004 |

### Cold-Start Precision@10

| History ratings | Strategy | Precision@10 |
| ---: | --- | ---: |
| 0 | popularity | 0.011 |
| 1 | content | 0.007 |
| 1 | popularity | 0.010 |
| 3 | content | 0.008 |
| 3 | popularity | 0.000 |
| 5 | matrix_factorization | 0.002 |
| 20 | matrix_factorization | 0.004 |

Each user is held out from global training and reveals only the listed number of their training ratings. The strategy column shows the policy actually used; a 1–4 rating user without a liked item falls back to popularity.

### Plots

![Matrix factorization training and validation loss](mf-loss.png)

![Precision@10 by revealed history size](cold-start-precision.png)

Naive Bayes accuracy is measured against D3 cluster labels. It reached 0.759 accuracy versus the 0.103 majority-class baseline on a held-out 20% of documents; training accuracy is 0.972. The earlier held-out accuracy was 0.068 because 4,895 of 5,183 cluster labels were paired with the wrong corpus documents: vector rows are sorted by document ID, while the loader preserved corpus order. These generated labels demonstrate classifier behavior, not real query intent.

Top training words for the first three topic IDs:

| Topic | Top 10 words by training frequency |
| ---: | --- |
| 15 | with (2165), 0 (1893), for (1595), were (1332), 1 (1317), was (1209), patients (1134), or (978), 2 (870), 95 (793) |
| 13 | cells (838), that (788), cell (684), cancer (664), is (598), by (594), with (530), tumor (472), we (451), for (427) |
| 11 | for (872), is (707), that (620), with (553), are (483), as (450), this (401), be (380), on (377), by (352) |
