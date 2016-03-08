# Compare Estimators

This project compares the performance of post-Lasso (BCCH, 2012) and RJIVE (Hansen and Kozbur 2014) with REL and BC-REL
in a linear IV model.

* The estimation of REL and BC-REL has been done.
* We need to code up post-Lasso and RJIVE


### Functionality

* `master_IV.m` is the master file for the Monte Carlo simulation.
* `dgpLinearIV.m` generates the data `y`, `x` and `z` in each simulation.
* `post_lasso.m` implements the post-lasso(BCCH, 2012) in which `LassoShooting2.m` does the pre-selection of instruments and `tsls.m` implements two-stage-least-square estimation. `prepareArgs.m` and `process_options` are supportive functions to drive `LassoShooting2.m`.
* `RJIVE.m` implements RJIVE (Hansen and Kozbur, 2014).
* `output_bias_rmse.m` computes the bias and RMSE for the estimation result.
* The results are saved in `report_bias_rmse.xls` and `result.mat`.