# Change log

# v0.2.3

- each growth model is now a separate class
- new growth model: Baranyi-Roberts with v=r (nu is free)
- fit to full data instead of just weights
- quantifiedcode integration 
- guessing nu gives false results, set it to 1
- fix D calculation in `lrtest` to be numerically stable
- refactor `calc_weights`
- added more competition models
- residuals plots much improved, added model residulas plot
- sample y0 in `compete`
- bootstrap sampling of model parameters
- `fit_model` accepts name of fitting method
- new module for likelihood analysis
- read and write csv files in Curveball format
- allow `none` blank in CLI - this avoids substracting the OD of the blank well from the other wells
- added weighted AIC and BIC to model attributes
- docs improved
- removed some tests: tests should check that code works, not that it gives positive results

# v0.2.2

- warning when number of samples from `sample_params` is lower than requested
- added Baranyi-Roberts model with nu=1 and v=r (see Baty & Delignette-Muller, 2004)
- reorganized parameter guessing and setting - closes #35
- `v=inf` when there is no lag phase
- added arguments to override parameters in competitions
- `plot_residuals` to plot the residuals of a model fit
- added arguments to `fit_model` and options to the CLI to control minimal values for parameters and to fix parameters to initial guess.

# v0.2.1

- added guess, param_max, and weights/no-weights as options to `curveball analyse` - closes #76
- fix max param settings in `curveball.models.fit_model`.
- hide future warnings in CLI when verbose is off - closes #100
- output RMSD, NRMSD, CV(RMSD) from `curveball analyse` - closes #94