# UMestimator

Code for reproducing the numerical experiments in the paper:
"Entropy estimation via normalizing flow".

## How to run the numerical experiments

The files are summarized as follows:

File                    | Experiments
------------------------|-----------------------------------------------------------------------------------------
`demo_mvn.py`           | Multivariate normal distribution, RMSE vs dimension
`demo_mvn_cr.py`        | Multivariate normal distribution, RMSE vs sample size
`demo_hr.py`            | Multivariate Rosenbrock distribution, RMSE vs dimension
`demo_hr_cr.py`         | Multivariate Rosenbrock distribution, RMSE vs sample size
`demo_lv_max_ent_ed.py` | Experimental design for the Lotka-Volterra model
`lv_val.py`             | Compute “reference value” of the entropy for the optimal observation time placements

To run the experiment of multivariate normal distribution in Fig. 2 (RMSE vs dimension), you can issue the command:
```
python demo_mvn.py
```

To run the experiment of multivariate normal distribution in Fig. 3 (RMSE vs sample size), you can issue the command:
```
python demo_mvn_cr.py
```

To run the experiments of multivariate Rosenbrock distribution, you can issue the command:
```
python demo_hr.py <mdl_name> <n_trials> 
```
for the figure of RMSE vs dimension, and issue the command:
```
python demo_hr_cr.py <mdl_name> <n_trials> 
```
for the figure of RMSE vs sample size. Here `<mdl_name>` is the name of distribution tested, and can be `hybrid_rosenbrock` or `even_rosenbrock`. `<n_trials> ` is the number of repeated trials and can be any positive integer.

To run the experiment of the optimal experimental design, you can issue the command:
```
python demo_lv_max_ent_ed.py <method> <n_samples>
```
Here, `<method>` is the entropy estimator used, and can be any of `kl`, `ksg`, `umtkl` , `umtksg` and `nf`. `<n_samples>` is the sample size, and can be any positive even number.

To compute the “reference value” of the optimal observation time placements obtained in this paper, you can issue the command:
```
python lv_val.py
```
