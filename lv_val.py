import numpy as np
from demo_lv_max_ent_ed_2 import util_fun

n_rep = 20
n_samples = 10000

n_grids = 20
xs = np.linspace(0.01, 0.99, n_grids)
ys = np.linspace(0.01, 0.99, n_grids)

u_tkl = np.empty((1,n_rep))
u_tksg = np.empty((1,n_rep))
u_kl = np.empty((1,n_rep))
u_ksg = np.empty((1,n_rep))
u_nf = np.empty((1,n_rep))

for i in range(20):    
    u_tkl[0,i] = util_fun(xs[7-1], ys[9-1], 'umtkl', n_samples)
    np.savetxt('temp_data/u_tkl', u_tkl) 
    u_tksg[0,i] = util_fun(xs[7-1], ys[9-1], 'umtksg', n_samples)
    np.savetxt('temp_data/u_tksg', u_tksg) 
    u_kl[0,i] = util_fun(xs[2-1], ys[4-1], 'kl', n_samples)
    np.savetxt('temp_data/u_kl', u_kl) 
    u_ksg[0,i] = util_fun(xs[4-1], ys[12-1], 'ksg', n_samples)
    np.savetxt('temp_data/u_ksg', u_ksg) 
    u_nf[0,i] = util_fun(xs[6-1], ys[8-1], 'nf', n_samples)
    np.savetxt('temp_data/u_nf', u_nf) 