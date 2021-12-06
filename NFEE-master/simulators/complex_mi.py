import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

from pdfs.gaussian import Gaussian

class mvn:
    
    def __init__(self, rho=0.8, dim_x=1):
        dim_y = dim_x
        self.x_dim = dim_x+dim_y
        self.rho = rho
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.sigma11 = np.eye(dim_x)
        self.sigma12 = rho*np.eye(dim_x,dim_y)
        self.sigma21 = self.sigma12.T
        self.sigma22 = np.eye(dim_y)
        self.S = np.concatenate((np.concatenate((self.sigma11, self.sigma12), axis=1),
                                 np.concatenate((self.sigma21, self.sigma22), axis=1)),axis=0)
        
        self.gaussian = Gaussian(m=np.ones(self.x_dim), S=self.S)
        
    def sim(self, n_samples=1000, rng=np.random):
        
        samples = self.gaussian.gen(n_samples=n_samples, rng=rng)
        return samples
    
    def plot_scatter(self, samples=None, n_samples=1000):
        import seaborn as sns
        import pandas as pd
        samples = self.sim(n_samples=n_samples)
        g = sns.PairGrid(pd.DataFrame(samples))
        g = g.map_diag(sns.kdeplot, shade=True, color='b', lw=3, legend=False)
        g = g.map_upper(plt.scatter)
        g = g.map_lower(sns.kdeplot)
        
    def density_fun(self, xs):
        return self.gaussian.eval(xs, log=False)
    
    def mcmc_entropy(self, n_samples=1000):
        return 0.5*np.log(np.linalg.det(2*math.pi*np.exp(1)*self.S))
        
    def mi(self, n_samples=1000, rng=np.random):
        return self.gaussian.kl(Gaussian(m=np.ones(self.x_dim), S=np.eye(self.x_dim)))



class mvquad:
    '''
    Even Rosenbrock distribution
    '''
    
    def __init__(self, rho=0.2, dim_x=1):
        self.x_dim = 2*dim_x
        self.rho = rho
        self.dim_x = dim_x
        self.dim_y = dim_x
        
    def sim(self, n_samples=1000, rng=np.random):
        
        xs = rng.randn(n_samples, self.dim_x)
        ys = xs**2 + self.rho*rng.randn(n_samples, self.dim_x)
        samples = np.concatenate([xs,ys], axis=1)
        return samples
    
    def plot_scatter(self, samples=None, n_samples=1000):
        import seaborn as sns
        import pandas as pd
        samples = self.sim(n_samples=n_samples)
        g = sns.PairGrid(pd.DataFrame(samples))
        g = g.map_diag(sns.kdeplot, shade=True, color='b', lw=3, legend=False)
        g = g.map_upper(plt.scatter)
        g = g.map_lower(sns.kdeplot)
        
    def density_fun(self, xs):
        
        dens = stats.multivariate_normal.pdf(xs[:, :self.dim_x], mean=np.zeros(self.dim_x), cov=np.eye(self.dim_x))*\
                stats.multivariate_normal.pdf(xs[:, self.dim_x:]-xs[:,:self.dim_x]**2, mean=np.zeros(self.dim_x), cov=self.rho**2*np.eye(self.dim_x))

        return dens
    
    def mcmc_entropy(self, n_samples=1000):
        
        #return -np.mean(np.log(self.density_fun(self.sim(n_samples))))
        return 0.5*np.log(np.linalg.det(2*math.pi*np.exp(1)*np.eye(self.dim_x)))+0.5*np.log(np.linalg.det(2*math.pi*np.exp(1)*self.rho**2*np.eye(self.dim_x)))
        
    def mi(self, n_samples=1000, rng=np.random):
        
        xs = rng.randn(n_samples, 1)
        ys = xs**2 + self.rho*rng.randn(n_samples, 1)
        
        x_ent = 0.5*np.log(np.linalg.det(2*math.pi*np.exp(1)*np.eye(1)))
        
        p_y = np.empty(n_samples)
        for i in range(n_samples):
            p_yx_sum = 0
            for j in range(n_samples):
                p_yx_sum = p_yx_sum + 1.0/n_samples*stats.norm.pdf(ys[i]-xs[j]**2, 0, self.rho) 
            
            p_y[i] = p_yx_sum
            
        y_ent = -np.mean(np.log(p_y))
        
        xy_ent = self.mcmc_entropy(n_samples)
        return self.dim_x*(x_ent+y_ent) - xy_ent
        

class mvquad2:
    '''
    Even Rosenbrock distribution
    '''
    
    def __init__(self, rho=0.2, dim_x=1):
        self.x_dim = 2*dim_x
        self.rho = rho
        self.dim_x = dim_x
        self.dim_y = dim_x
        
    def sim(self, n_samples=1000, rng=np.random):
        
        xs = 1+rng.randn(n_samples, self.dim_x)
        ys = xs**2 + self.rho*rng.randn(n_samples, self.dim_x)
        samples = np.concatenate([xs,ys], axis=1)
        return samples
    
    def plot_scatter(self, samples=None, n_samples=1000):
        import seaborn as sns
        import pandas as pd
        samples = self.sim(n_samples=n_samples)
        g = sns.PairGrid(pd.DataFrame(samples))
        g = g.map_diag(sns.kdeplot, shade=True, color='b', lw=3, legend=False)
        g = g.map_upper(plt.scatter)
        g = g.map_lower(sns.kdeplot)
        
    def density_fun(self, xs):
        
        dens = stats.multivariate_normal.pdf(xs[:, :self.dim_x], mean=np.ones(self.dim_x), cov=np.eye(self.dim_x))*\
                stats.multivariate_normal.pdf(xs[:, self.dim_x:]-xs[:,:self.dim_x]**2, mean=np.zeros(self.dim_x), cov=self.rho**2*np.eye(self.dim_x))

        return dens
    
    def mcmc_entropy(self, n_samples=1000):
        
        #return -np.mean(np.log(self.density_fun(self.sim(n_samples))))
        return 0.5*np.log(np.linalg.det(2*math.pi*np.exp(1)*np.eye(self.dim_x)))+0.5*np.log(np.linalg.det(2*math.pi*np.exp(1)*self.rho**2*np.eye(self.dim_x)))
        
    def mi(self, n_samples=1000, rng=np.random):
        
        xs = rng.randn(n_samples, 1)
        ys = xs**2 + self.rho*rng.randn(n_samples, 1)
        
        x_ent = 0.5*np.log(np.linalg.det(2*math.pi*np.exp(1)*np.eye(1)))
        
        p_y = np.empty(n_samples)
        for i in range(n_samples):
            p_yx_sum = 0
            for j in range(n_samples):
                p_yx_sum = p_yx_sum + 1.0/n_samples*stats.norm.pdf(ys[i]-xs[j]**2, 0, self.rho) 
            
            p_y[i] = p_yx_sum
            
        y_ent = -np.mean(np.log(p_y))
        
        xy_ent = self.mcmc_entropy(n_samples)
        return self.dim_x*(x_ent+y_ent) - xy_ent       