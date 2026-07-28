# PRE-R10 supporting descriptive audit

- Implementation: `WGT-V11-R9-SUPPORTING-DESCRIPTIVE-v1.0.1-result-aware-authorized-pre-R10`
- Analysis status: `PRE_R10_SUPPORTING_DESCRIPTIVE_RESULT_AWARE__NO_NEW_CONFIRMATORY_INFERENCE`
- Input: the immutable `-02` R9 event export only.
- Excluded: `-01`, E3, old R9 v1, manuscript Results writing, and R10.
- Confirmatory R9 v2 remains unchanged and authoritative for its 30 hypotheses.
- This audit adds no stochastic confidence interval, p-value, sign-flip test,
  Holm family, or C4 decision.
- Failure direction: comparator rate minus proposed rate; positive favors the
  proposed method because failure is lower-is-better.
- Hard-violation bounds set every missing execution observation to no violation
  for the lower endpoint bound and to violation for the upper endpoint bound.
  Method-related missingness blocks a complete method-comparison statement.
- Cost ratios are proposed/comparator. Three available sets are reported:
  every completed task pair, equal charged-work pairs, and pairs where both
  methods accepted every event. No set is retrospectively promoted to a
  confirmatory cost gate.
- R10 remains blocked. These artifacts are inputs for a future separately
  authorized writing stage, not manuscript prose.
