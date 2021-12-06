import numpy as np
from scipy import stats

from ent_est.entropy import kl, ksg, tkl, tksg
from simulators.complex import mvn

d = 40
n_trials = 100
sim_mdl = mvn(rho=0.0, dim_x=d)
all_n_samples = [5000, 2*5000, 3*5000, 4*5000, 5*5000, 6*5000]

plt_xscale_log = True
plt_yscale_log = True

mse1 = np.empty(len(all_n_samples))
mse2 = np.empty(len(all_n_samples))
mse3 = np.empty(len(all_n_samples))
mse4 = np.empty(len(all_n_samples))
        
true_val = sim_mdl.mcmc_entropy()

for n, n_samples in enumerate(all_n_samples):
        
    cal1 = np.empty(n_trials)
    cal2 = np.empty(n_trials)
    cal3 = np.empty(n_trials)
    cal4 = np.empty(n_trials)
                
    for i in range(n_trials):
            
        n_sims = n_samples
        cal1[i] = kl(sim_mdl.sim(n_samples=n_sims))
        cal2[i] = ksg(sim_mdl.sim(n_samples=n_sims))
                
        xs = sim_mdl.sim(n_samples=n_sims)
        ys = stats.norm.cdf(xs)
        cal3[i] = tkl(ys) - np.mean(np.log(np.prod(stats.norm.pdf(xs), axis=1)))
                
        xs = sim_mdl.sim(n_samples=n_sims)
        ys = stats.norm.cdf(xs)
        cal4[i] = tksg(ys) - np.mean(np.log(np.prod(stats.norm.pdf(xs), axis=1)))
                
    mse1[n] = np.mean((cal1-true_val)**2)
    mse2[n] = np.mean((cal2-true_val)**2)
    mse3[n] = np.mean((cal3-true_val)**2)
    mse4[n] = np.mean((cal4-true_val)**2)


import matplotlib.pyplot as plt
plt.switch_backend('agg')
            
fig, ax = plt.subplots(1,1)
            
ax.plot(all_n_samples, np.sqrt(mse3), marker='o', color='b', linestyle=':', label='UM-tKL', mfc='none')
ax.plot(all_n_samples, np.sqrt(mse4), marker='o', color='b', linestyle='-', label='UM-tKSG', mfc='none')    
ax.plot(all_n_samples, np.sqrt(mse1), marker='x', color='r', linestyle=':', label='KL')
ax.plot(all_n_samples, np.sqrt(mse2), marker='x', color='r', linestyle='-', label='KSG')

if plt_xscale_log == True:
    ax.set_xscale("log")
    
if plt_yscale_log == True:
    ax.set_yscale("log")
        
ax.set_xlabel('N')
ax.set_ylabel('RMSE')
ax.legend()
plt.savefig('figs/RMSE_vs_N_mvn')


import util.io
import os
util.io.save((all_n_samples, mse1, mse2, mse3, mse4), os.path.join('temp_data', 'RMSE_vs_N_mvn')) 