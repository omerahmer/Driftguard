-- Driftguard initial schema.
--
-- This mirrors the agreed data model 1:1. Notes on non-obvious choices:
--
--  * UUID primary keys via gen_random_uuid() (built into Postgres 13+, so no
--    extension needed). IDs travel through CLI args, --json output, PR comments
--    and the GitHub Action, so non-guessable, globally-unique IDs are safer to
--    pass around than serials and don't leak row counts.
--
--  * EMBEDDINGS ARE DEFERRED. The data model has no vector columns, and a
--    pgvector column needs a fixed dimension (e.g. vector(1024)) dictated by the
--    embedding model — an intentionally open Phase 3 decision. We enable the
--    `vector` extension now (cheap, and the smoke test uses it) but add the
--    actual vector(N) columns in a Phase 3 migration once the model is chosen.
--
--  * Two foreign keys form a cycle (prompts.current_version_id <->
--    prompt_versions.prompt_id) and one is self-referential
--    (prompt_versions.parent_version_id). In raw SQL we control statement order,
--    so we create the tables first and attach those FKs with ALTER TABLE at the
--    end of this same migration — no separate migration needed.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE prompts (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name               text NOT NULL UNIQUE,
    -- The active version. Nullable + FK added below: the create flow is insert
    -- prompt (NULL) -> insert first version -> update prompt, because the cycle
    -- can't be satisfied in a single insert.
    current_version_id uuid,
    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE prompt_versions (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_id         uuid NOT NULL REFERENCES prompts (id) ON DELETE CASCADE,
    content           text NOT NULL,
    -- Not in the literal field list, but implied by diff_from_parent: a diff is
    -- meaningless without knowing what it is a diff *from*. Making it explicit
    -- lets us walk history as a chain. Nullable: the first version has no parent.
    parent_version_id uuid,
    -- Serialized diff vs. the parent version. Stored as text/JSON; the diff
    -- granularity (line/sentence/semantic) is a Phase 2 open decision and any
    -- chosen format serializes to text.
    diff_from_parent  text,
    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE eval_cases (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_id         uuid NOT NULL REFERENCES prompts (id) ON DELETE CASCADE,
    input             text NOT NULL,
    -- Natural-language description of a correct response. This is what we embed
    -- in Phase 3 and feed to the judge in Phase 4.
    expected_behavior text NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE eval_runs (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_version_id   uuid NOT NULL REFERENCES prompt_versions (id) ON DELETE CASCADE,
    eval_case_id        uuid NOT NULL REFERENCES eval_cases (id) ON DELETE CASCADE,
    actual_output       text NOT NULL,
    -- Judge verdict for "did this output satisfy expected_behavior". Nullable:
    -- a run can be recorded before it is judged.
    judge_passed        boolean,
    judge_justification text,
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE selection_records (
    id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_version_id     uuid NOT NULL REFERENCES prompt_versions (id) ON DELETE CASCADE,
    eval_case_id          uuid NOT NULL REFERENCES eval_cases (id) ON DELETE CASCADE,
    -- Continuous cosine similarity between the changed prompt span and the eval
    -- case's expected_behavior. double precision because Phase 4 sweeps
    -- thresholds over it; exact equality is never meaningful.
    similarity_score      double precision NOT NULL,
    -- Did our heuristic pick this case to run? (score >= threshold)
    was_selected          boolean NOT NULL,
    -- Ground truth from Phase 4: did behavior actually change? Compared against
    -- was_selected to compute precision/recall. Nullable until ground truth exists.
    was_actually_affected boolean,
    created_at            timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE behavior_diffs (
    id                       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_case_id             uuid NOT NULL REFERENCES eval_cases (id) ON DELETE CASCADE,
    prompt_version_before_id uuid NOT NULL REFERENCES prompt_versions (id) ON DELETE CASCADE,
    prompt_version_after_id  uuid NOT NULL REFERENCES prompt_versions (id) ON DELETE CASCADE,
    -- Judge verdict: did substantive behavior change before -> after? The
    -- ground-truth signal Phase 4 scores selection against.
    judge_behavior_changed   boolean,
    judge_justification      text,
    -- Human spot-check of the judge verdict (judge-validate command). Nullable:
    -- only a sampled subset is reviewed, and that agreement rate tells us whether
    -- to trust the judge as ground truth.
    human_agreed             boolean,
    created_at               timestamptz NOT NULL DEFAULT now()
);

-- Deferred cyclic / self-referential foreign keys (see header comment).
-- ON DELETE SET NULL so removing the active/parent version doesn't cascade-delete
-- the prompt or descendant versions; the application repoints the link instead.
ALTER TABLE prompts
    ADD CONSTRAINT prompts_current_version_id_fkey
    FOREIGN KEY (current_version_id) REFERENCES prompt_versions (id) ON DELETE SET NULL;

ALTER TABLE prompt_versions
    ADD CONSTRAINT prompt_versions_parent_version_id_fkey
    FOREIGN KEY (parent_version_id) REFERENCES prompt_versions (id) ON DELETE SET NULL;
