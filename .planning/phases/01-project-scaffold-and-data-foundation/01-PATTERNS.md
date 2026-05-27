# Phase 1: Project Scaffold and Data Foundation - Pattern Map

**Mapped:** 2026-05-27
**Status:** Complete

## Scope

There is no existing implementation code in this repository yet. Pattern mapping therefore uses the PRD and planning docs as the closest available analogs.

## Files To Create

| File | Role | Source Pattern |
|------|------|----------------|
| `.gitignore` | Prevent restricted/generated files from entering git | `AGENTS.md` dataset license constraint and `01-CONTEXT.md` D-04 |
| `requirements.txt` | Reproducible Python dependencies | `.planning/research/STACK.md` and `MLPC_Task4_PRD.md` |
| `src/__init__.py` | Importable Python package marker | Standard Python module layout |
| `src/config.py` | Central constants and paths | `MLPC_Task4_PRD.md` config block and `01-CONTEXT.md` D-02/D-11 |
| `src/data.py` | Metadata, feature, label, and cache logic | `MLPC_Task4_PRD.md` Phase 1 function boundaries |
| `tests/test_data.py` | Unit tests for deterministic data transforms | `01-VALIDATION.md` validation map |
| `results/log.md` | Human-readable verification notes | Requirement `SETUP-04` |

## Data Flow

`src/config.py` defines paths and constants. `src/data.py` reads `metadata.csv`, `annotations.csv`, and `audio_features/*.npz`, then writes `results/dataset_cache.npz` and appends/updates `results/log.md` with summary checks.

## Constraints

- Do not commit `data/`, raw audio, feature `.npz` files, generated caches, model binaries, or generated PDFs.
- Do not implement splits, preprocessing, metrics, baseline, training, report, or slides in Phase 1.
- Keep functions small and testable because Phase 2 and later scripts consume the cache contract.

## PATTERN MAPPING COMPLETE
