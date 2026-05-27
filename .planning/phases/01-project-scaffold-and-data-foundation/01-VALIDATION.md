---
phase: 01
slug: project-scaffold-and-data-foundation
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-27
---

# Phase 01 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pytest.ini` or default pytest discovery |
| **Quick run command** | `python -m pytest tests/test_data.py` |
| **Full suite command** | `python -m pytest` |
| **Estimated runtime** | < 10 seconds without the course dataset |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_data.py` once tests exist.
- **After every plan wave:** Run `python -m pytest`.
- **Before `$gsd-verify-work`:** Full suite must be green, and the dataset smoke command must run if `data/` is available.
- **Max feedback latency:** 10 seconds for unit tests; dataset smoke runtime depends on local dataset size.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01-01 | 1 | SETUP-01 | T-01-01 | Dataset and generated artifacts ignored by git | source | `git check-ignore data results/dataset_cache.npz` | no | pending |
| 01-01-02 | 01-01 | 1 | SETUP-02, SETUP-03, SETUP-04 | T-01-02 | Config avoids embedded secrets and absolute private paths | source | `python -m compileall src` | no | pending |
| 01-02-01 | 01-02 | 1 | DATA-01, DATA-02 | T-01-03 | Loader reads local-only dataset path and validates schema | unit | `python -m pytest tests/test_data.py` | no | pending |
| 01-03-01 | 01-03 | 2 | DATA-03, DATA-04 | T-01-04 | Missing annotators do not become negative votes | unit | `python -m pytest tests/test_data.py` | no | pending |
| 01-03-02 | 01-03 | 2 | DATA-05, DATA-06 | T-01-05 | Cache excludes raw audio and logs only summary stats | smoke | `python -m src.data` | no | pending |

---

## Wave 0 Requirements

- [ ] `tests/test_data.py` - unit tests for label aggregation and feature concatenation.
- [ ] `requirements.txt` - include `pytest` so validation can run locally.
- [ ] `src/__init__.py` - makes `src` importable during tests.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Course dataset path exists locally | DATA-01, DATA-05, DATA-06 | The licensed dataset is not committed and may be provided as a local directory or symlink | Confirm `data/metadata.csv`, `data/annotations.csv`, and `data/audio_features/*.npz` exist before running `python -m src.data` |

---

## Validation Sign-Off

- [x] All tasks have automated or smoke verification commands.
- [x] Sampling continuity avoids long unverified stretches.
- [x] Wave 0 covers missing test infrastructure.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 10 seconds for unit tests.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending
