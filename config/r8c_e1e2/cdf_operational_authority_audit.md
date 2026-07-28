# R8C E1+E2 CDF operational-authority and reference-front audit

- Audit ID: `WGT-V11-R8C-E1E2-CDF-OPERATIONAL-AUTHORITY-AUDIT-01`
- Amendment ID: `WGT-V11-R8C-E1E2-CDF-OPERATIONAL-AUTHORITY-AMENDMENT-01`
- Audit date: `2026-07-26`
- Status: result-blind corrective audit; not formal-execution authority
- Effect outputs inspected: no
- Observed method outputs used: no

## Frozen sources

1. Version-of-record paper: P. A. Grudniewski and A. J. Sobey,
   “Benchmarking the performance of genetic algorithms on constrained dynamic
   problems,” *Natural Computing* 21, 109–125 (2022), version of record
   2020-07-22, DOI
   [10.1007/s11047-020-09799-y](https://doi.org/10.1007/s11047-020-09799-y).
2. Author executable oracle:
   [Pag1c18/cmlsga](https://bitbucket.org/Pag1c18/cmlsga),
   commit `1926a5a1c89adf0a5e5e70449adbec62750a108a`, file
   `MLSGA/Fit_Functions.cpp`, 461,394 bytes, SHA-256
   `48b2c256f4bdec6ed4f81f8edd82a03753bc51550776e1ae84b2d6fcbc18fa7a`.
3. Historical Python binding:
   `src/benchmark_adapters/r4_evaluators.py`, class `CDFEvaluator`. It remains
   an immutable historical identity and is not silently relabelled.
4. Corrective Python binding:
   `src/benchmark_adapters/cdf_operational.py`, class
   `CDFOperationalEvaluator`, suite
   `CDF-1-15-CMLSGA-1926A5A1-OPERATIONAL`.

The paper and author code are not mutually consistent in every equation. The
minimal reproducible rule is therefore to use the named author commit as the
operational equation authority, retain the paper as the scientific
description, and enumerate every conflict below. This rule was selected
without inspecting any method effect output.

## Per-problem equation audit

“Match” means the objective and constraint equations relevant to execution
agree after accounting for indexing and the project-wide sign conversion
(`author g >= 0` becomes `project c <= 0`).

| Problem | Version-of-record paper | Author oracle commit | Historical Python | Corrective action |
|---|---|---|---|---|
| CDF1 | Places the `sign(k) sqrt(abs(k))` term inside each exponent. | Matches the paper. | Places that term outside the power, so the variable constraint differs. | Override CDF1 only, using the oracle exponent placement. |
| CDF2 | CF4-derived objective and one variable constraint. | Match. | Match. | Inherit unchanged. |
| CDF3 | UDF3 ripple and one variable constraint. | Match. | Match. | Inherit unchanged. |
| CDF4 | UDF objective and one objective-space sinusoidal constraint. | Match. | Match. | Inherit unchanged. |
| CDF5 | Shifted multimodal objective and one variable constraint. | Match. | Match. | Inherit equations; correct the reference front for decision-bound reachability. |
| CDF6 | Writes `-|G(t)|` in both variable constraints. | Omits those two shifts. | Matches the oracle omission. | Freeze the oracle behavior and document the paper conflict. |
| CDF7 | Shifted UDF2 objective and CF1 constraint. | Match. | Match. | Inherit unchanged. |
| CDF8 | UDF5-derived ideal curve and CF3 objective constraint. | Match. | Match. | Inherit equations; reference only the feasible subset of the stated ideal curve. |
| CDF9 | Writes `-|G(t)|` in both variable constraints and a real square root of `q = 1-(M x1)^H`. | Omits the two shifts; evaluates the same real square root without defining behavior for `q < 0`. | Matches the oracle equations and historically raises a generic numerical error when `q < 0`. | Keep the oracle equations. Raise typed `CDFDomainUndefinedError` for `q < 0`; do not clamp, sign-extend, narrow the frozen bounds, resample, or invent an objective value. |
| CDF10 | Squares `w_j`, and its displayed constraint does not match the implemented CF6 pair. | Sums `w_j` directly and implements two CF6 variable constraints. | Matches the oracle. | Freeze the oracle objective and both constraints. |
| CDF11 | Adds `|G(t)|` to both objectives and uses `2 y^2-cos(4 pi y)+1`. | Has no objective shift and uses `y^2-cos(4 pi y)+1`. | Matches the oracle. | Freeze the oracle equations; derive the finite front after enforcing the `[-1,1]` decision bounds. |
| CDF12 | Uses cosine for odd-index linkage variables and sine for even-index variables. | Uses sine for odd indices and cosine for even indices. | Matches the oracle. | Freeze the oracle sine/cosine ordering. |
| CDF13 | Uses `K(t1)` but does not define it completely in Eq. 13. | Defines `K(t1) = ceil(10 G(t1))`. | Matches the oracle. | Freeze the oracle definition and bind every reference identity to profile, event, seed, complete five-component time vector, and evaluator hash. |
| CDF14 | Dynamic CF1 constraint. | Match. | Match. | Inherit unchanged; classify exact zero-shift events by rational event arithmetic, not floating `sin(n pi)`. |
| CDF15 | Writes `sin(2 pi A + G(t))`. | Implements `sin(2 pi (A + G(t)))`. | Matches the oracle. | Freeze the oracle parenthesization and derive its feasible intervals analytically. |

## Compact true-front and extrema rules

The normalization catalog does not store 10,000 sampled points per
problem-event. Continuous fronts store only objective-wise extrema and a
small derivation/root certificate. A finite front stores every unique true
point in lexicographic order.

Let `g = sin(pi t/2)`, `a = |g|`, and `m = 0.5+a`.

### CDF5: decision-bound reachable lower envelope

For fixed `x=x1`, all linkage residuals except the constrained `x2` residual
can be zero within `[-2,2]`. Define

```text
b(x,g) = 0.8 x sin(6 pi x + 0.2 pi) + g
A(x)   = 0.5 x - 0.25
I(x,g) = [max(-2-b(x,g), A(x)), 2-b(x,g)]
T      = 1.5 (1-sqrt(2)/2)
w(r)   = |r|                         for r < T
         0.125 + (r-1)^2             for r >= T
W5(x,g)= min_{r in I(x,g)} w(r)
F2(x,g)= 1-x+a+W5(x,g)
```

`W5` is obtained from the interval endpoints and the reachable stationary
points `r=0` and `r=1`. The Pareto set consists of record lows of `F2` as
`x` increases. Consequently, the normalization extrema are

```text
minimum = (a, min_x F2(x,g))
maximum = (a+x*, 1+a)
```

where `x*` is the earliest global minimizer. The implementation performs a
deterministic global scan of this piecewise analytic envelope followed by
local refinement. This corrects the author plotting routine when `r=1` is
not reachable because `x2` would exceed its upper bound.

### CDF8: feasible subset of the ideal curve

On the ideal curve,

```text
f1=x
f2=1-m x^m
c8(x)=sqrt(x)-m x^m-sin(2 pi (sqrt(x)+m x^m)).
```

Only `x` satisfying `c8(x) >= 0` belongs to the true front. The smallest and
largest feasible `x` are deterministically bracketed; the extrema are
`(xmin, 1-m xmax^m)` and `(xmax, 1-m xmin^m)`.

### CDF9: real domain, reachable CF6 branches, and typed failure

Define `q(x)=1-(m x)^m`. The real source equation exists only for
`x <= min(1,1/m)`. Within that domain, satisfying the two variable
constraints with minimum reachable residual cost gives

```text
R(q) = q^2          for q >= 0.5
       0.5 q        for 0.25 <= q < 0.5
       0.25 sqrt(q) for 0 <= q < 0.25.
```

Thus the front is `(x+a, R(q(x))+a)` over the real domain. For `q < 0`, the
candidate is charged exactly once on scalar entry, the ledger records
`CDFDomainUndefinedError`, and the frozen external terminal semantics remain
`REJECT_NUMERICAL`. A natural batch containing such a row must fail before
any batch ledger mutation and then follow the existing ordered scalar
fallback. No automatic retry is introduced.

### CDF11: complete bound-reachable finite front

The oracle ripple is `h(x)=0.15 |sin(pi(20x+g))|`. The only nondominated
candidate abscissae are the in-bound ripple zeros
`x=(k-g)/20`, plus the two endpoints before dominance filtering. At each
candidate the constrained `x2` residual minimizes the same piecewise `w`
over

```text
[max(-1-0.8 x sin(6 pi x+0.2 pi), 0.5x-0.25),
  1-0.8 x sin(6 pi x+0.2 pi)].
```

The complete true front is the unique nondominated subset of
`(x+h(x), 1-x+W11(x)+h(x))`. The catalog stores every such point. Dense
independent domination checks are tests only; their samples are not stored.

### CDF13: seed- and five-time-component-bound feasible subset

Let `g_i=sin(pi t_i/2)`, `a=|g_3|`, `M=0.5+|g_4|`, and
`H=0.5+|g_5|`. The ideal curve is

```text
f1=x+a
f2=1-M x^H+a
c13(x)=a+M((x+a)^H-x^H)
         -sin(2 pi (M((x+a)^H+x^H)-a)).
```

Only `c13(x) >= 0` is retained. The smallest and largest feasible `x` are
bracketed. Each identity includes the profile, event, master seed, exact
five-component time vector, and evaluator SHA-256; the amendment separately
binds the frozen schedule commitments for the five seeds.
`K(t1)=ceil(10 G(t1))` affects the reachable decision witness and is frozen
even though it cancels from objective-space extrema.

### CDF15: analytic feasible intervals

For the oracle parenthesization, the ideal constraint reduces to

```text
-sin(2 pi (2x^2+g)) >= 0.
```

All integer intervals `2x^2+g in [k+0.5,k+1]`, intersected with `x in
[0,1]`, are enumerated exactly. If their global endpoints are `xmin,xmax`,
the extrema are `(xmin,1-xmax^2)` and `(xmax,1-xmin^2)`.

## Validation and evidence boundary

- Fifteen independent known-answer vectors are frozen against the operational
  equations.
- Both profiles and all 60 events are checked for ordered scalar/batch
  equality.
- The first five frozen dynamic seeds are checked for the CDF13 schedule and
  one-component-per-event update.
- CDF5 and CDF11 decision-bound reachability is checked explicitly.
- CDF8, CDF13, and CDF15 front endpoints are evaluated through constructed
  decision witnesses.
- CDF9 checks scalar charging, original failure typing, zero-side-effect batch
  fallback, and the unchanged `REJECT_NUMERICAL` external terminal.
- Reference identities and artifacts are self-hashed; the JSONL file is
  independently hash- and line-count-bound by its manifest.
- Formal R9 analysis must fail closed if the amendment, evaluator, audit
  document, catalog manifest, catalog file, or schema hash differs.

This audit changes no algorithm, seed, sample count, endpoint, CFE budget, or
research question. It does not authorize formal execution, effect analysis,
or results writing.
