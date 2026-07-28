# R9 compact readable outputs

This directory is a derived, human-readable view of one immutable raw R8C
E1+E2 root. `r9_export_manifest.json` binds every file here by byte count and
SHA-256. It contains no narrative effect conclusion.

- `task_endpoints.csv`: one row per scheduled sequence. Empty endpoint cells
  mean not computed; `endpoint_status` gives the reason. Values are never
  imputed for failed or numerically excluded tasks.
- `e2_negative_transfer.csv`: one row per frozen E2 comparator/FULL pair.
  `pair_status` identifies included and unavailable pairs.
- `failure_cost.csv`: one row per scheduled sequence, including failed and
  partial tasks. Unknown charged work remains empty and
  `charged_work_exact` remains false. Authenticated hard-kill JSONL tail
  fragments are reported by presence, byte count and SHA-256; they are never
  interpreted as completed events.
- `post_execution_hard_violation.csv`: one row per scheduled sequence.
  Its status and observation counts preserve task failure or missing
  execution observations; a rate is emitted only for a complete denominator.
- `event_diagnostics.jsonl.gz`: optional and absent by default. When requested,
  it contains one row per durably completed event, including authenticated
  prefixes from partial tasks.

The raw root is not modified or deleted by this export.
