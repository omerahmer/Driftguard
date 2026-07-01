-- Phase 3: add the deferred pgvector embedding columns.
--
-- Dimension 1024 is fixed by the chosen embedding model, Voyage AI's
-- voyage-3.5-lite (configurable via Matryoshka to 256/512/1024/2048; we use
-- 1024). This is exactly the decision Phase 1 deferred — a pgvector column needs
-- a concrete dimension, and that dimension is dictated by the model. If the model
-- ever changes, this becomes a new migration with a new dimension, and existing
-- embeddings are recomputed.
--
-- Both columns are nullable: embeddings are computed lazily (on first
-- `select-evals`) after the row already exists, then cached in place.
--
-- The `vector` extension was already enabled in 0001.

-- The embedding of a version's CHANGED SPAN (the added-side text of its diff
-- from parent). This is the "query" side of the similarity comparison.
ALTER TABLE prompt_versions
    ADD COLUMN changed_span_embedding vector(1024);

-- The embedding of each eval case's expected_behavior. The "document" side.
ALTER TABLE eval_cases
    ADD COLUMN expected_behavior_embedding vector(1024);

-- No ANN index (HNSW/IVFFlat) on purpose: at this project's scale (hundreds of
-- eval cases) an exact sequential scan is fast and, unlike an approximate index,
-- returns exact cosine scores — which matters because Phase 4 measures
-- precision/recall against those scores. An index would trade exactness for a
-- throughput we don't need yet.
