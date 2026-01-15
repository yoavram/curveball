# Curveball Update Plan (Minimal Change)

## Goals
- Make Curveball installable in modern Python environments via `pip`.
- Update dependency constraints without changing public APIs.
- Keep code changes limited to I/O and packaging metadata.

## Steps
1. **Remove conda environment file**
   - Delete `environment.yml` and update docs to reference `pip`-based installs.

2. **Relax dependency pins**
   - Update `setup.py` and `requirements.txt` to modern, compatible ranges.
   - Replace the strict `xlrd==1` pin with `xlrd>=1.2.0,<2.0` to preserve `.xlsx` support without forcing a legacy build.
   - Add explicit `python-dateutil` (used in `curveball/ioutils.py`).

3. **Documentation touch-up**
   - Update `docs/install.rst` to remove references to `environment.yml` and keep pip-first guidance.

4. **Validation**
   - Run `pytest -v tests` (or `make test`) and fix only compatibility issues.

## Non-Goals
- No API redesigns or refactors.
- No new features beyond dependency compatibility fixes.
