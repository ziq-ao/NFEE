import argparse

from simulators.complex import Hybrid_Rosenbrock
from simulators.complex_mi import mvquad
from ent_est import entropy



def plot_RMSE_vs_d(mdl_name, n_trials):
    
    import numpy as np

    if mdl_name == 'hybrid_rosenbrock':
        all_sigma = [4, 7, 10, 13, 16, 19, 22]
        all_params = [1, 2, 3, 4, 5, 6, 7]
        para_model = lambda sigma: Hybrid_Rosenbrock(mu=1, n_1=4, n_2=sigma, a=1, b=0.1)
        n_hiddens=[50,50]
        act_fun='tanh'
        n_mades=10
        plt_xscale_log = False
        plt_yscale_log = False
        
    elif mdl_name == 'even_rosenbrock':
        all_sigma = [4, 6, 8, 10, 14, 18, 22]
        all_params = [2, 3, 4, 5, 7, 9, 11]
        para_model = lambda sigma: mvquad(rho=0.2, dim_x=sigma)
        n_hiddens=[50,50]
        act_fun='tanh'
        n_mades=5
        plt_xscale_log = False
        plt_yscale_log = False
        
        
    all_model = []
    
    true_val = np.empty_like(all_sigma)
    
    all_n_samples = [100, 500]
    
    mse1 = np.empty((len(all_n_samples), len(all_sigma)))
    mse2 = np.empty((len(all_n_samples), len(all_sigma)))
    mse3 = np.empty((len(all_n_samples), len(all_sigma)))
    mse4 = np.empty((len(all_n_samples), len(all_sigma)))
    mse5 = np.empty((len(all_n_samples), len(all_sigma)))
    
    
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
            
    for k, sigma in enumerate(all_sigma):
    
        sim_model = para_model(all_params[k])
        all_model.append(sim_model)
            
        true_val =  sim_model.mcmc_entropy(1000000)
        
        for n, n_samples in enumerate(all_n_samples):
            
            cal1 = np.empty(n_trials)
            cal2 = np.empty(n_trials)
            cal3 = np.empty(n_trials)
            cal4 = np.empty(n_trials)
            cal5 = np.empty(n_trials)
    
            for i in range(n_trials):
                
                net = create_model(sim_model.x_dim, rng=np.random) 
                estimator = entropy.UMestimator(sim_model, net) 
                estimator.learn_transformation(n_samples=n_samples*sim_model.x_dim/2)
            
                cal1[i], cal5[i], _, _ = estimator.calc_ent(reuse_samples=False, method='umtkl')
                cal2[i], cal5[i], _, _ = estimator.calc_ent(reuse_samples=False, method='umtksg')
                
                estimator.n_samples = n_samples*sim_model.x_dim
                cal3[i] = estimator.ksg_ent(reuse_samples=False, method='kl')
                cal4[i] = estimator.ksg_ent(reuse_samples=False, method='ksg')
            
            mse1[n,k] = np.mean((cal1-true_val)**2)
            mse2[n,k] = np.mean((cal2-true_val)**2)
            mse3[n,k] = np.mean((cal3-true_val)**2)
            mse4[n,k] = np.mean((cal4-true_val)**2)
            mse5[n,k] = np.mean((cal5-true_val)**2)
        
    all_mse = [mse1,mse2,mse3,mse4,mse5]
    
    import numpy as np
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    
    fig0, ax0 = plt.subplots(1,1)
    
    ax0.plot(all_sigma, np.sqrt(mse1[0]), label='UM-tKL', color='b', linestyle=':')
    ax0.plot(all_sigma, np.sqrt(mse2[0]), label='UM-tKSG', color='b', linestyle='--')
    ax0.plot(all_sigma, np.sqrt(mse3[0]), label='KL', color='r', linestyle=':')
    ax0.plot(all_sigma, np.sqrt(mse4[0]), label='KSG', color='r', linestyle='--')
    ax0.plot(all_sigma, np.sqrt(mse5[0]), label='NF', color='g', linestyle='--')
    
    if plt_xscale_log == True:
        ax0.set_xscale("log")
    
    if plt_yscale_log == True:
        ax0.set_yscale("log")
    
    ax0.set_xlabel('dimension')
    ax0.set_ylabel('RMSE')
    ax0.legend()
    fig0.savefig('figs/RMSE_vs_d_{0}_{1}d'.format(mdl_name, all_n_samples[0]))
    
    fig1, ax1 = plt.subplots(1,1)
    
    ax1.plot(all_sigma, np.sqrt(mse1[1]), label='UM-tKL', color='b', linestyle=':')
    ax1.plot(all_sigma, np.sqrt(mse2[1]), label='UM-tKSG', color='b', linestyle='--')
    ax1.plot(all_sigma, np.sqrt(mse3[1]), label='KL', color='r', linestyle=':')
    ax1.plot(all_sigma, np.sqrt(mse4[1]), label='KSG', color='r', linestyle='--')
    ax1.plot(all_sigma, np.sqrt(mse5[1]), label='NF', color='g', linestyle='--')
    
    if plt_xscale_log == True:
        ax1.set_xscale("log")
    
    if plt_yscale_log == True:
        ax1.set_yscale("log")
    
    ax1.set_xlabel('dimension')
    ax1.set_ylabel('RMSE')
    ax1.legend()
    fig1.savefig('figs/RMSE_vs_d_{0}_{1}d'.format(mdl_name, all_n_samples[1]))
    
    import util.io
    import os
    util.io.save((all_sigma, all_mse), os.path.join('temp_data', 'RMSE_vs_d_'+mdl_name)) 
    
    
def main():

    parser = argparse.ArgumentParser(description='Plot the RMSE vs d for Rosenbrock models.')
    parser.add_argument('mdl_name', type=str, help='the name of the model')
    parser.add_argument('n_trials', type=int, help='the number of repeated trials')
    args = parser.parse_args()

    plot_RMSE_vs_d(args.mdl_name, args.n_trials)
    
if __name__ == '__main__':
    main()