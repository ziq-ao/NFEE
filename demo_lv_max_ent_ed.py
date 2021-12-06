import numpy as np
from simulators.time_series import LV2
from scipy import stats
import argparse

from ent_est import entropy

n_sensors = 5
sim_model = LV2(n_sensors=n_sensors)

def beta_scheme(a, b):
    
    a = np.exp(0.8*stats.norm.ppf(a))
    b = np.exp(0.8*stats.norm.ppf(b))
    beta = stats.beta(a, b)
    t = beta.ppf(1.0/(n_sensors+1)*np.linspace(1, n_sensors, n_sensors))
    
    return t

def create_measure_time(t):
    
    return (sim_model.tmax/sim_model.timestep*t).astype(int)


def create_model(n_inputs, rng):
    
        import ml.models.mafs as mafs
    
        return mafs.MaskedAutoregressiveFlow(
                    n_inputs=n_inputs,
                    n_hiddens=[50, 50],
                    act_fun='tanh',
                    n_mades=5,
                    mode='random',
                    rng=rng
                )



def util_fun(d1, d2, method, n_samples):
    
    sim_model.measure_time = create_measure_time(beta_scheme(d1, d2))
    
    if method == 'kl':
        
        return entropy.kl(sim_model.sim(n_samples))
    
    elif method == 'ksg':
        
        return entropy.ksg(sim_model.sim(n_samples))
    
    elif method == 'umtkl' or method == 'umtksg': 
        
        net = create_model(sim_model.x_dim, rng=np.random)
        estimator = entropy.UMestimator(sim_model, net)
        data_train = sim_model.sim(n_samples=n_samples/2)
        estimator.samples = data_train
        estimator.learn_transformation(n_samples=n_samples/2)
        data_comp = sim_model.sim(n_samples=n_samples/2)
        estimator.samples = data_comp
        h, _, _, _ = estimator.calc_ent(reuse_samples=True, method=method)
        return h
    
    elif method == 'nf': 
        
        net = create_model(sim_model.x_dim, rng=np.random)
        estimator = entropy.UMestimator(sim_model, net)
        data_train = sim_model.sim(n_samples=n_samples/2)
        estimator.samples = data_train
        estimator.learn_transformation(n_samples=n_samples/2)
        data_comp = sim_model.sim(n_samples=n_samples/2)
        estimator.samples = data_comp
        _, h, _, _ = estimator.calc_ent(reuse_samples=True, method='umtksg')
        return h

    
def gs_ed(method, n_samples):
    '''
    grid search experimental design
    '''
    
    n_grids = 20
    xs = np.linspace(0.01, 0.99, n_grids)
    ys = np.linspace(0.01, 0.99, n_grids)
    u = np.zeros((n_grids, n_grids))
    for i in range(n_grids):
        for j in range(n_grids):
            u[i, j] = util_fun(xs[i], ys[j], method, n_samples)
            np.savetxt('temp_data/lv_max_ent_{0}_{1}'.format(method, n_samples), u)    
            
    return u
    
    
    
def main():

    parser = argparse.ArgumentParser(description='Experiental design for maximum entropy for LV model.')
    parser.add_argument('method', type=str, choices=['kl', 'ksg', 'umtkl', 'umtksg', 'nf'], help='the method of entropy estimation')
    parser.add_argument('n_samples', type=int, help='the sample size used')
    args = parser.parse_args()

    gs_ed(args.method, args.n_samples)
    
if __name__ == '__main__':
    main()
