import argparse

from simulators.complex import Hybrid_Rosenbrock
from simulators.complex_mi import mvquad
from ent_est import entropy



def plot_RMSE_vs_N(mdl_name, n_trials):
    
    import numpy as np
    # model settings
    if mdl_name == 'hybrid_rosenbrock':
        d = 3
        sim_model = Hybrid_Rosenbrock(mu=1, n_1=4, n_2=d, a=1, b=0.1)
        all_n_samples = [2**10, 2**11, 2**12, 2**13, 2**14, 2**15]
        n_mades=10
        
    elif mdl_name == 'even_rosenbrock':
        d = 5
        sim_model = mvquad(rho=0.2, dim_x=d)
        all_n_samples = [2**10, 2**11, 2**12, 2**13, 2**14, 2**15]
        n_mades=5
    
    # net settings
    n_hiddens=[50,50]
    act_fun='tanh'
    
    plt_xscale_log = True
    plt_yscale_log = True
    
    
    mse1 = np.empty(len(all_n_samples))
    mse2 = np.empty(len(all_n_samples))
    mse3 = np.empty(len(all_n_samples))
    mse4 = np.empty(len(all_n_samples))
    mse5 = np.empty(len(all_n_samples))
    
    true_val = sim_model.mcmc_entropy(1000000)
    
    def create_model(n_inputs, rng):
        
            import ml.models.mafs as mafs
        
            return mafs.MaskedAutoregressiveFlow(
                        n_inputs=n_inputs,
                        n_hiddens=n_hiddens,
                        act_fun=act_fun,
                        n_mades=n_mades,
                        mode='random',
                        rng=rng
                    )
        
    for n, n_samples in enumerate(all_n_samples):
            
        cal1 = np.empty(n_trials)
        cal2 = np.empty(n_trials)
        cal3 = np.empty(n_trials)
        cal4 = np.empty(n_trials)
        cal5 = np.empty(n_trials)
            
        for i in range(n_trials):
                
            net = create_model(sim_model.x_dim, rng=np.random) 
            estimator = entropy.UMestimator(sim_model, net) 
            estimator.learn_transformation(n_samples=n_samples/2)
            
            cal1[i], cal5[i], _, _ = estimator.calc_ent(reuse_samples=False, method='umtkl')
            cal2[i], cal5[i], _, _ = estimator.calc_ent(reuse_samples=False, method='umtksg')
                
            estimator.n_samples = n_samples
            cal3[i] = estimator.ksg_ent(reuse_samples=False, method='kl')
            cal4[i] = estimator.ksg_ent(reuse_samples=False, method='ksg')
            
        mse1[n] = np.mean((cal1-true_val)**2)
        mse2[n] = np.mean((cal2-true_val)**2)
        mse3[n] = np.mean((cal3-true_val)**2)
        mse4[n] = np.mean((cal4-true_val)**2)
        mse5[n] = np.mean((cal5-true_val)**2)
        
    all_mse = [mse1,mse2,mse3,mse4,mse5]
    
    import numpy as np
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    
    fig, ax = plt.subplots(1,1)
    ax.plot(all_n_samples, np.sqrt(mse1), label='UM-tKL', color='b', linestyle=':')
    ax.plot(all_n_samples, np.sqrt(mse2), label='UM-tKSG', color='b', linestyle='--')
    ax.plot(all_n_samples, np.sqrt(mse3), label='kl', color='r', linestyle=':')
    ax.plot(all_n_samples, np.sqrt(mse4), label='ksg', color='r', linestyle='-')
    ax.plot(all_n_samples, np.sqrt(mse5), label='NF', color='g', linestyle='--')
    
    if plt_xscale_log == True:
        ax.set_xscale("log")
    
    if plt_yscale_log == True:
        ax.set_yscale("log")
    
    ax.set_xlabel('N')
    ax.set_ylabel('RMSE')
    ax.legend()
    
    fig.savefig('figs/RMSE_vs_N_{0}'.format(mdl_name))
    
    import util.io
    import os
    util.io.save((all_n_samples, all_mse), os.path.join('temp_data', 'RMSE_vs_N_'+mdl_name)) 
    
def main():

    parser = argparse.ArgumentParser(description='Plot the RMSE vs N for Rosenbrock models.')
    parser.add_argument('mdl_name', type=str, help='the name of the model')
    parser.add_argument('n_trials', type=int, help='the number of repeated trials')
    args = parser.parse_args()

    plot_RMSE_vs_N(args.mdl_name, args.n_trials)
    
if __name__ == '__main__':
    main()