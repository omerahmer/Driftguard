-- Hand-labeled ground truth for the eval selector.
--
-- behavior_diffs.judge_behavior_changed is the LLM judge's verdict — cheap and
-- scalable, but it's an LLM grading an LLM. This column is an INDEPENDENT human
-- label of the same question ("did behavior substantively change before ->
-- after?"), captured by the labeling flow (CLI `judge-validate` / the dashboard
-- labeling view).
--
-- With it we can (a) score the selector's precision/recall against gold human
-- ground truth, and (b) audit the judge by comparing judge_behavior_changed to
-- this column (treating the human label as truth). Nullable: only labeled rows
-- count toward either use.
--
-- This supersedes the earlier human_agreed boolean (agreement with the judge).
-- human_agreed is left in place for backward compatibility but is no longer
-- written; agreement is now derived (human_behavior_changed = judge_behavior_changed).
ALTER TABLE behavior_diffs
    ADD COLUMN human_behavior_changed boolean;
