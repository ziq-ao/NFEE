import numpy as np
from scipy import stats

from ent_est.entropy import kl, ksg, tkl, tksg
from simulators.complex import mvn

ds = [1,10,20,30,40]
n_trials = 100
all_n_samples = [1000]

mse1 = np.empty((len(all_n_samples), len(ds)))
mse2 = np.empty((len(all_n_samples), len(ds)))
mse3 = np.empty((len(all_n_samples), len(ds)))
mse4 = np.empty((len(all_n_samples), len(ds)))
        
for k, d in enumerate(ds):
    
    sim_mdl = mvn(rho=0.0, dim_x=d)
    
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
                
        true_val = sim_mdl.mcmc_entropy()
        mse1[n,k] = np.mean((cal1-true_val)**2)
        mse2[n,k] = np.mean((cal2-true_val)**2)
        mse3[n,k] = np.mean((cal3-true_val)**2)
        mse4[n,k] = np.mean((cal4-true_val)**2)
    
import matplotlib.pyplot as plt
plt.switch_backend('agg')
            
fig, ax = plt.subplots(1,1)
            
ax.plot(ds, np.sqrt(mse3[0]), marker='o', color='b', linestyle=':', label='UM-tKL', mfc='none')
ax.plot(ds, np.sqrt(mse4[0]), marker='o', color='b', linestyle='-', label='UM-tKSG', mfc='none')    
ax.plot(ds, np.sqrt(mse1[0]), marker='x', color='r', linestyle=':', label='KL')
ax.plot(ds, np.sqrt(mse2[0]), marker='x', color='r', linestyle='-', label='KSG')
        
ax.set_xlabel('dimension')
ax.set_ylabel('RMSE')
ax.legend()
plt.savefig('figs/RMSE_vs_d_mvn')
        
import util.io
import os
util.io.save((ds, mse1, mse2, mse3, mse4), os.path.join('temp_data', 'RMSE_vs_d_mvn')) 