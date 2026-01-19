# Repository Guidelines

## Project Structure & Module Organization
The core library lives in `curveball/` (models, competitions, I/O, plots, CLI). Tests are in `tests/` and use `test_*.py` naming. Documentation sources are under `docs/` and built with Sphinx. Example datasets are in `data/` and example scripts/notebooks in `examples/`. Plate templates are stored in `plate_templates/`. Build artifacts and packaging metadata appear in `build/`, `dist/`, and `curveball.egg-info/`.

## Build, Test, and Development Commands
- `pip install -U -e .[tests]`: install the package in editable mode with test dependencies.
- `make test`: run the test suite with coverage (generates `coverage_report/`).
- `pytest -v tests`: run tests directly without coverage.
- `make doc`: build HTML docs and open `docs/_build/html/index.html`. When using pixi, run the same target through `pixi run make doc`.
- `tox`: run tests across supported Python versions (if tox is configured locally).

## Coding Style & Naming Conventions
Python code follows standard PEP 8 conventions: 4‑space indentation, `snake_case` for functions/variables, and `CamelCase` for classes. Keep APIs and module-level functions descriptive and aligned with existing names in `curveball.models` and `curveball.competitions`. There is no required formatter or linter configured in this repo.

## Testing Guidelines
Tests are executed with `pytest` and live in `tests/`. Coverage is expected to be at least 80% when running `make test` (see coverage options in `Makefile`). Name new tests `test_<feature>.py`, and use descriptive test function names (e.g., `test_fit_model_logistic()`).

## Commit & Pull Request Guidelines
There is no strict commit message format in the history; use concise, imperative summaries (e.g., “Fix competition ODE edge case”). Avoid `[skip ci]` in normal commits. PRs should include a clear description, reference relevant issues, and note any new dependencies or data files. Include plots or screenshots when changes affect visualization outputs.

## Documentation & Release Notes
Documentation is Sphinx-based; edit `docs/*.rst` and rebuild with `make doc`. For releases, update `CHANGELOG.md` with a short summary of changes and ensure version tags follow `v#.#.#` (PEP 440).
