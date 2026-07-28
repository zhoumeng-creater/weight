# DT-RAMDE Journal Code and Processed Evidence

This repository contains the public journal implementation and processed
evidence for:

> **Dual-Time-Scale Rejection-Aware Memory Differential Evolution for
> Dynamic and Receding-Horizon Constrained Multiobjective Optimization**

DT-RAMDE is a constrained multiobjective differential-evolution method with
separate within-event and cross-event adaptation. The implementation records
rejection-aware credit, dual time-scale parameter memories, guarded memory
reset, and receding-horizon execution feedback. The repository supports
algorithmic and synthetic benchmark claims only; it is not a clinical
decision system and contains no participant-level clinical data.

## Authors

- **Meng Zhou** — Hainan International College, Communication University of
  China, Binhai Campus, Li'an International Education Innovation Pilot Zone,
  Lingshui, Hainan 572426, China.
  Email: zhoumeng@cuc.edu.cn ·
  [ORCID 0009-0007-2559-9901](https://orcid.org/0009-0007-2559-9901)
- **Danting Duan** (corresponding author) — Department of Electrical and
  Electronic Engineering, Hanyang University ERICA, Ansan, Republic of Korea.
  Email: dantingduan.cn@gmail.com

The earlier conference implementation is preserved unchanged on the
[`conference-paper`](https://github.com/zhoumeng-creater/weight/tree/conference-paper)
branch. The `main` branch is the journal-code line.

## Repository contents

- `src/` — DT-RAMDE, evaluation, benchmark, formal-execution, and analysis
  packages.
- `tools/` — fail-closed runners, validators, exporters, and R9 analysis
  entrypoints.
- `config/` — versioned machine contracts, schemas, benchmark records, and
  frozen provenance bindings.
- `tests/` — correctness, contract, qualification, and regression tests.
- `data/processed/event-export-v1/` — processed formal-run export used by R9.
- `data/processed/inference-v2/` — confirmatory inference tables.
- `data/processed/supporting-v2/` — supporting failure, hard-violation, and
  computational-cost tables.

`config/r8c_e1e2/cdf_operational_authority_audit.md` is retained because its
scientific derivation is hash-bound by the CDF evaluator, reference catalog,
schemas, and regression tests. It is part of the executable evidence chain,
not an internal planning record.

## Installation

Python 3.12 is required. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test,qualification,benchmark]"
```

For strict reproduction on the frozen Linux x86_64 target, install the
SHA-256-bound environment before installing this repository without resolving
additional dependencies:

```bash
python -m pip install -r requirements-r8c-linux-x86_64.lock
python -m pip install --no-deps -e .
```

Run the complete test suite:

```powershell
python -m pytest -q
python -m ruff check src tests tools
```

Run a small public correctness fixture. The output path must be an absolute,
previously nonexistent directory outside the repository:

```powershell
$smokeOutput = Join-Path ([System.IO.Path]::GetTempPath()) "dt-ramde-smoke-$(Get-Date -Format yyyyMMddHHmmssfff)"
python tools/run_v11_experiment.py --fixture static_bridge_e0 --scope public_correctness_fixture --seed 2026 --output-root $smokeOutput
```

## Reproduce the R9 analyses

The packaged R9 implementation contracts are content-addressed. Run these
commands from the repository root after installation.

Confirmatory inference:

```powershell
$inferenceOutput = Join-Path ([System.IO.Path]::GetTempPath()) "dt-ramde-r9-inference-$(Get-Date -Format yyyyMMddHHmmssfff)"
python tools/run_v11_r9_inference.py `
  --input-root data/processed/event-export-v1 `
  --r5-contract config/r5/r5_freeze_contract.json `
  --implementation-contract config/r9/r9_inference_implementation_v1_0_1.json `
  --implementation-contract-sha256 78684bfd9871690cbae8840eecfc2c6106aece9072e50428aa2a8773c1d56f26 `
  --output-root $inferenceOutput `
  --authorize-r9-inference R9_RESULT_AWARE_CONFIRMATORY_INFERENCE_AUTHORIZED
```

Supporting descriptive analysis:

```powershell
$supportingOutput = Join-Path ([System.IO.Path]::GetTempPath()) "dt-ramde-r9-supporting-$(Get-Date -Format yyyyMMddHHmmssfff)"
python tools/run_v11_r9_supporting_descriptive.py `
  --input-root data/processed/event-export-v1 `
  --r5-contract config/r5/r5_freeze_contract.json `
  --implementation-contract config/r9/r9_supporting_descriptive_implementation_v1_0_1.json `
  --implementation-contract-sha256 aa8dbf88a6b395c994b75769d42690f475cba088f9ec4fd8aa3adfd04f7d1f78 `
  --output-root $supportingOutput `
  --authorize-pre-r10-supporting-audit PRE_R10_FAST_PUBLICATION_ROUTE_AUTHORIZED_NO_R10
```

The generated CSV files should be byte-identical to the corresponding files
under `data/processed/inference-v2/` and
`data/processed/supporting-v2/`. Generated manifests record the current public
contract hashes and therefore intentionally differ from manifests created by
the private formal-run workspace.

## Provenance and data availability

The formal execution that produced the processed export was bound to:

- source commit: `29ad459b8b898f967c04005c413446f641e14424`
- source tree: `dfb046c2200ac5f1a9a5a14dcb64f4f02796476b`
- raw run-manifest SHA-256:
  `33ab590adf809ca2b1f87c1ef225a18d43f50dbc40d7f3c2e2da7a379b1768d3`

The public source snapshot retains the scientific implementation and processed
evidence while replacing private, one-time authorization text and machine-local
paths with stable public-release tokens and repository-relative paths. No
endpoints, samples, thresholds, algorithms, seeds, or numerical result tables
were changed by that sanitization.

The processed analysis inputs and outputs needed to reproduce the reported R9
tables are included in this repository. The complete raw formal-run directory
is not stored in Git because of its size; it may be requested from the
corresponding author. No archival data DOI has yet been assigned.

## Licensing

Project-authored repository contents are released under the
[MIT License](LICENSE). Third-party components retain their own licenses:
jMetalPy is installed as an MIT-licensed dependency, while the GPL-licensed CDF
author implementation was used only as an external provenance oracle and is
not copied, linked, or redistributed here. See
`config/r4/license_record.json` for the machine-readable source and license
record.
