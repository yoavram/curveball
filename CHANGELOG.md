# Change log

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