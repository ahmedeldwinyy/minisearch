# Project status (paused Oct 2026)

## Done
- Phase A: foundation
- Phase B: tokenizer, inverted index, boolean search, query parser, trie, benchmarks
- Phase C: TF-IDF, BM25, metrics (BM25 NDCG@10 = 0.667)
- Phase D: code complete

## Next (do first)
Phase D fix pass, partly done:
- DONE on main: cluster labels aligned with document vectors (Naive Bayes bug); recommenders ranked with raw MF scores
- ON BRANCH wip/phase-d-fix (135 tests pass, lint and mypy pass, numbers not reviewed): updated recommender report, plot, eval script and test
- STILL TO DO: re-check Naive Bayes accuracy vs the 0.103 majority baseline; re-check MF precision@10 vs popularity (0.010); confirm the gradient check passes; tune MF on a validation split; add an oracle baseline

## After that
Phase E (packaging, async ingestion), F (PostgreSQL, FastAPI), G (testing), H (Docker, CI), I (system design), J (publish)

## Rules
One brick at a time, test first, make test and make lint must pass, one commit per brick, push after each.
