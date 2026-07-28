# R9 versioned confirmatory inference

- Implementation: `WGT-V11-R9-INFERENCE-v1.0.1-result-aware-authorized`
- Implementation contract SHA-256: `78684bfd9871690cbae8840eecfc2c6106aece9072e50428aa2a8773c1d56f26`
- R5 contract SHA-256: `4e2dd0a0f4a97b57d71dd13eb60aa8a3c3eb34f0708aae609d50a31d155f6554`
- R9 event export manifest SHA-256: `9b9761360294b6194aea05d09504223c49fafee26f9e343e5fd7e5667d0b9e94`
- Raw run manifest SHA-256: `33ab590adf809ca2b1f87c1ef225a18d43f50dbc40d7f3c2e2da7a379b1768d3`
- Scope: E1 and E2 only; E3 is excluded.
- Direction: proposed minus comparator for both higher-is-better endpoints.
- CI: paired stratified hierarchical cluster bootstrap, 20,000 replicates,
  95% linear percentile interval, frozen PCG64 seed. Rolling instances are
  resampled within each fixed template; paired seeds are resampled within
  each selected cluster and fixed profile/template stratum.
- Test: two-sided paired top-level cluster sign-flip, 100,000 replicates,
  plus-one p-value, frozen PCG64 seed.
- Multiplicity: Holm step-down within each of the three frozen families,
  familywise alpha 0.05.
- Missing numerical continuous endpoints: no imputation in confirmatory tests;
  available-case estimates are accompanied by endpoint-[0,1] FAS bounds.
- RNG consumption: one stream per procedure, consumed sequentially in the
  30-row canonical hypothesis order.
- Provenance: this implementation was created after effect visibility under
  explicit author authorization; it does not claim to be a result-blind
  pre-specified software implementation.

`confirmatory_hypotheses.csv` contains the 30 registered comparisons.
`top_level_cluster_effects.csv` contains the raw independent-unit effects and
missing-data bounds used by the inference.
