# Change log

## v0.2.16

- Docs: use curl instead of wget, update figures and tables
- Force xlrd v1 because v2 does not support xlsx
- Bumped some dependency versions
- Fixed small minor bugs

## v0.2.15

- Bug fixes

## v0.2.8

- Removed central estimation in `curveball.models.*_ci` functions.
- Build with conda-build version 1; version 2 caused problems on travis-ci (see [PR #1342](https://github.com/conda/conda-build/pull/1342)).
- Removed python-dateutil dependency.

## v0.2.7

- Fixed bug: `find_K_ci` was called with wrong args in `cli.py`.
- Added `--nsamples` to CLI
- Test with CLI with `--ci`
- Adjusted to [seaborn API change](https://github.com/mwaskom/seaborn/commit/69e7f371d27725160d092a528c96cf1fce99b8b4): `sns.Grid`->`sns.axisgrid.Grid`

## v0.2.6

- added tests for `competitions.py`
- fixed bad docstrings
- avoid loading qt
- add `y0` arg to competition CI function
- update to lmfit 0.9.3 to solve bug
- confidence interval version for `fit_and_compete`
- fix bootstrap bug
- raise dev status to alpha in PyPI identifiers
- plot sampled curves and sampled model fits
- small bug fixes

See commit log for more details

## v0.2.5

- When using a `find_*_ci` function, it now also returns an estimate - the average of all samples (bug fix). 
- Added `find_K_ci` to find the confidence interval of the K parameter (closes #131)

## v0.2.4

- Bug fixes
- Fit and compete - fit total OD to total of competition model (699f8b6)
- Lotka Volterra competition models (started at fbc6dc8)
- New competitions model based on resource consumption (22d949c)
- Smoothing is more stable and doesn't require parameters (97c64d4)
- Fix definition of `q0` in docs
- Measure minimal doubling time (7ffdd24; suggested by Idan Frumkin)
- Measure confidence interval for max growth rate, lag, and min doubling time and report in CLI (d95bf3d and further commits afterwards)
- Improved parameter guess functions (97c64d4, 0cecfaf)
- Fit exponential model (523b30270ff6cd95d6702ed1b497e3f12655129b)
- Support for Python 3.5 (4a1c5b8)
- Moved docs from divshot to netlify (fbe022d)

## v0.2.3

- each growth model is now a separate class
- new growth model: Baranyi-Roberts with v=r (nu is free)
- fit to full data instead of just weights
- quantifiedcode integration 
- guessing nu gives false results, set it to 1
- fix D calculation in `lrtest` to be numerically stable
- refactor `calc_weights`
- added more competition models
- residuals plots much improved, added model residuals plot
- sample y0 in `compete`
- bootstrap sampling of model parameters
- `fit_model` accepts name of fitting method
- new module for likelihood analysis
- read and write csv files in Curveball format
- allow `none` blank in CLI - this avoids subtracting the OD of the blank well from the other wells
- added weighted AIC and BIC to model attributes
- docs improved
- removed some tests: tests should check that code works, not that it gives positive results

## v0.2.2

- warning when number of samples from `sample_params` is lower than requested
- added Baranyi-Roberts model with nu=1 and v=r (see Baty & Delignette-Muller, 2004)
- reorganized parameter guessing and setting - closes #35
- `v=inf` when there is no lag phase
- added arguments to override parameters in competitions
- `plot_residuals` to plot the residuals of a model fit
- added arguments to `fit_model` and options to the CLI to control minimal values for parameters and to fix parameters to initial guess.

## v0.2.1

- added guess, param_max, and weights/no-weights as options to `curveball analyse` - closes #76
- fix max param settings in `curveball.models.fit_model`.
- hide future warnings in CLI when verbose is off - closes #100
- output RMSD, NRMSD, CV(RMSD) from `curveball analyse` - closes #94
