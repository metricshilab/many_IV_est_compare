# Estimation in the Linear IV Model with Many IVs

#### Zhentao Shi

This project compares the finite-sample performance of post-Lasso (Belloni, Chen, Chernozhukov and Hansen, 2012), RJIVE (Hansen and Kozbur, 2014),
and RLMIL (Carrasco and Tchuente, 2015) in the linear IV model with many IVs.

Description of the data generating process and implementation of the estimators can be find in

Zhento Shi (2017): ["Econometric Estimation with High-Dimensional Moment Equalities"](http://www.sciencedirect.com/science/article/pii/S0304407616301373), *Journal of Econometrics*

### Contributors
* [Eddie Gao](https://github.com/Eddie-Gao), my research assistant, develops the code under my supervision.
* I design the DGP, and develop the code of [REL and BC-REL](http://zhentaoshi.github.io/REL/) for comparison, as in the paper.
* We thank G.Tchuente for sharing his RLIML code.
* Belloni, Chen, Chernozhukov and Hansen share the post-Lasso code with their published paper.


### Functionality

The code is written in Matlab.

* `master_IV.m` is the master file for the simulation.
* `dgpLinearIV.m` generates the data `y`, `x` and `z` in each simulation.
* `post_lasso.m` implements the post-lasso (BCCH, 2012).
    * `LassoShooting2.m` does the pre-selection of instruments.
    * `tsls.m` implements two-stage-least-square estimation.
    * `prepareArgs.m` and `process_options` are supportive functions to drive `LassoShooting2.m`.
* `RJIVE.m` implements RJIVE (Hansen and Kozbur, 2014).
* `Rliml.m` implements regularized LIML estimation under Tikhonov regularization (Carrasco and Tchuente, 2015).
* `output_bias_rmse.m` computes the bias and RMSE for the estimation result.
