# Run Versions Log (`/eos/user/y/yeo/4l/runs/<version>`)

`v1/v2/v3/...` are **run-attempt versions**: each new attempt gets a fresh
`RUNS_ROOT` so a failed/partial attempt's outputs never mix with the next.
Condor logs are grouped the same way under `condor/logs/<version>/`.

Set the version in `pipeline/categories.py` (`RUNS_ROOT`), then regenerate +
submit. **Bump to a new version whenever you re-run after a fix** — don't reuse a
version that already holds a failed attempt.

Only versions that were **actually submitted** are recorded here, with evidence
of what happened.

Status legend: ❌ failed · ⚠️ partial

| ver | date | change from previous | outcome | root cause |
|-----|------|----------------------|---------|-----------|
| v1 | 2026-06-25 | first full submission (resnet50, epochs 100) | ❌ failed | GPU `cudaErrorNoKernelImageForDevice` — jobs landed on V100/V100S (cc 7.0); LCG torch needs cc ≥ 7.5 |
| v2 | ~2026-06-29 | + `require_gpus (Capability≥7.5)` | ❌ failed | (a) `KeyError: effective_pt_bin_edges` at efficiency step; (b) EOS **FUSE** errors: `Level 2 not synchronized` (write), `can't retrieve stat info` (read), some `Permission denied` |
| v3 | 2026-06-30 | + `MY.SendCredential` + `effective_pt_bin_edges` attrs in weight files | ⚠️ mostly failed (2/30 models) | SendCredential fixed the `Permission denied` class, but `Level 2 not synchronized` (FUSE **write**) persisted — jobs still died at `output_dir.mkdir()` |

---

## Details per version

### v1 — first full submission ❌
- Config: 18 image (resnet50) + 12 track, epochs 100, `request_gpus=1` with **no GPU-capability constraint**.
- Failure: first GPU op → `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`. Warning: `Found GPU0 Tesla V100S ... cuda capability 7.0 ... Minimum ... (7.5)`.
- Root cause: cvmfs `LCG_109_cuda` PyTorch supports CUDA capability **7.5–9.0**; V100/V100S are **7.0**, and nothing excluded them.
- Fix applied for next attempt: `require_gpus = (Capability >= 7.5)` — commit `c2954e3`.

### v2 — GPU capability constrained ❌
- Change: `require_gpus = (Capability >= 7.5)` → V100/V100S avoided. GPU error gone.
- New failures:
  1. **`KeyError: 'effective_pt_bin_edges'`** at the efficiency-vs-pT step (`train_resnet:1114`, `track:470`, `fusion:579`). The object-based category weight builder never wrote that top-level attr, so jobs trained fine but crashed **before** saving predictions/metrics → no usable outputs.
  2. **EOS FUSE errors**: `[Errno 45] Level 2 not synchronized` on output `mkdir`, `can't retrieve stat info` reading input HDF5, some `Permission denied`.
- Fixes applied for next attempt: write `requested/effective_pt_bin_edges` attrs (added to the 6 existing weight files + builder — commit `f21073e`); forward Kerberos credential (commit `d551538`).

### v3 — effective_pt_bin_edges + SendCredential ⚠️
- Changes: `MY.SendCredential = True` (forward Kerberos to workers for EOS auth) + `effective_pt_bin_edges` present in weight files.
- Result: **only 2/30 jobs produced `best_model.pt`.** SendCredential resolved the `Permission denied` (credential) class, but **`Level 2 not synchronized` (FUSE write) persisted** — most jobs still died at `output_dir.mkdir()` on the EOS FUSE mount.
- Lesson recorded: a trivial `kerberos_test` mkdir job (5/5 OK) was wrongly generalized to "EOS fixed"; it only exercised the credential class, not FUSE-write reliability. See skill `evidence-before-claims`.
