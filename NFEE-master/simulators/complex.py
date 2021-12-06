import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import scipy.special as spl
import math

from pdfs.gaussian import Gaussian
from pdfs.mog import MoG
from util.plot import plot_hist_marginals

class Rosenbrock:
    
    def __init__(self, sigma):
        
        self.x_dim = 2
        self.sigma = sigma
    
    def sim(self, n_samples=1000, rng=np.random):
        
        x = rng.randn(n_samples,1)
        y = x**2 + self.sigma * rng.randn(n_samples,1)
        
        samples = np.concatenate((x,y), axis=1)
        
        return samples
    
    def plot_scatter(self, samples=None, n_samples=1000):
        
        if samples is None:
            samples = self.sim(n_samples)
        
        fig, ax = plt.subplots(1,1)
        ax.scatter(samples[:,0], samples[:,1] , c='b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        return fig
        
    def density_fun(self, xs):
        
        xs = np.asarray(xs)
        if xs.ndim == 1:
            xs = xs[np.newaxis,:]
        
        dens = stats.norm.pdf(xs[:,0])*stats.norm.pdf(xs[:,1]-xs[:,0]**2, scale=self.sigma)
            
        return dens
    
    def mcmc_entropy(self, n_samples=1000):
        
        return -np.mean(np.log(self.density_fun(self.sim(n_samples))))
        
     
        
class roll:
    
    def __init__(self, sigma, T):
        
        self.x_dim = 3
        self.sigma = sigma
        self.T = T
    
    def sim(self, n_samples=1000, rng=np.random):
        
        x = rng.randn(n_samples,1)
        y = np.cos(x/self.T) + self.sigma * rng.randn(n_samples,1)
        z = np.sin(x/self.T) + self.sigma * rng.randn(n_samples,1)
        
        samples = np.concatenate((x,y,z), axis=1)
        
        return samples
    
    def plot_scatter(self, samples=None, n_samples=1000):
        
        if samples is None:
            samples = self.sim(n_samples)
        
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(samples[:,0], samples[:,1], samples[:,2] , c='b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        return fig
        
    def density_fun(self, xs):
        
        xs = np.asarray(xs)
        if xs.ndim == 1:
            xs = xs[np.newaxis,:]
        
        dens = stats.norm.pdf(xs[:,0])*\
                stats.norm.pdf(xs[:,1]-np.cos(xs[:,0]/self.T), scale=self.sigma)*\
                stats.norm.pdf(xs[:,2]-np.sin(xs[:,0]/self.T), scale=self.sigma)
            
        return dens
    
    def mcmc_entropy(self, n_samples=1000):
        
        return -np.mean(np.log(self.density_fun(self.sim(n_samples))))
        

class five_d_quadratic:
    
    def __init__(self, sigma):
        
        self.x_dim = 5
        self.sigma = sigma
    
    def sim(self, n_samples=1000, rng=np.random):
        
        x = rng.rand(n_samples, 4)-0.5
        y = np.sum(x**2, axis=1)[:, np.newaxis] + self.sigma*(rng.rand(n_samples,1)-0.5)
        
        samples = np.concatenate((x,y), axis=1)
        
        return samples
        
    def density_fun(self, xs):
        
        xs = np.asarray(xs)
        if xs.ndim == 1:
            xs = xs[np.newaxis,:]
        
        dens = 1/self.sigma*np.ones(xs.shape[0])
            
        return dens
    
    def mcmc_entropy(self, n_samples=1000):
        
        return -np.mean(np.log(self.density_fun(self.sim(n_samples))))
    

class mvn:
    
    def __init__(self, rho=0.8, dim_x=1):
        self.x_dim = dim_x
        self.rho = rho
        self.dim_x = dim_x
        self.S = rho*np.ones((dim_x, dim_x))+(1-rho)*np.eye(dim_x)     
        self.gaussian = Gaussian(m=np.zeros(self.x_dim), S=self.S)
        
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
    
class mog:
    
    def __init__(self, rho=0.8, dim_x=1):
        
        self.x_dim = dim_x
        self.rho = rho
        self.dim_x = dim_x
        self.m1 = 5*np.ones(dim_x)
        self.S1 = rho*np.ones((dim_x, dim_x))+(1-rho)*np.eye(dim_x)   
        self.m2 = np.zeros(dim_x)
        self.S2 = (1-rho)*np.eye(dim_x)  
        self.S2[0, 0] = 1.0
        self.mog = MoG(a=[0.5, 0.5], ms=[self.m1, self.m2], Ss=[self.S1, self.S2])
        
    def sim(self, n_samples=1000, rng=np.random):
        
        samples = self.mog.gen(n_samples=n_samples, rng=rng)
        return samples
    
    def plot_scatter(self, samples=None, n_samples=1000):

        samples = self.sim(n_samples=n_samples)
        plot_hist_marginals(samples)
    
    def density_fun(self, xs):
        return self.mog.eval(xs, log=False)
    
    def mcmc_entropy(self, n_samples=1000):
        return -np.mean(np.log(self.density_fun(self.sim(n_samples))))

class n_d_beta:
    
    def __init__(self, a, b, dim_x=1):
        self.x_dim = dim_x
        self.a = a
        self.b = b
        self.rv = stats.beta(a, b)
        
    def sim(self, n_samples=1000, rng=np.random):
        
        samples = self.rv.rvs(size=[n_samples, self.x_dim])
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
        return np.prod(self.rv.pdf(xs), axis=1)
    
    def mcmc_entropy(self, n_samples=1000):
        return self.x_dim*(np.log(spl.beta(self.a, self.b))-(self.a-1)*spl.digamma(self.a)\
                -(self.b-1)*spl.digamma(self.b)+(self.a+self.b-2)*spl.digamma(self.a+self.b))
        #return -np.mean(np.log(self.density_fun(self.sim(n_samples))))  
    
class n_d_quadratic:
    
    def __init__(self, sigma):
        
        self.noise = 0.1
        self.x_dim = sigma
        self.sigma = sigma
    
    def sim(self, n_samples=1000, rng=np.random):
        
        x = rng.randn(n_samples, self.sigma-1)
        y = np.sum(x**2, axis=1)[:, np.newaxis] + self.noise*(rng.randn(n_samples,1))
        
        samples = np.concatenate((x,y), axis=1)
        
        return samples
        
    def density_fun(self, xs):
        
        xs = np.asarray(xs)
        if xs.ndim == 1:
            xs = xs[np.newaxis,:]
        
        dens = stats.multivariate_normal.pdf(xs[:,:self.sigma-1], mean=np.zeros(self.sigma-1), cov=np.eye(self.sigma-1))*\
                stats.norm.pdf(xs[:,-1]-np.sum(xs[:,:self.sigma-1]**2, axis=1), scale=self.noise)
            
        return dens
    
    def mcmc_entropy(self, n_samples=1000):
        
        return -np.mean(np.log(self.density_fun(self.sim(n_samples))))


class n_d_unif:
    
    def __init__(self, sigma):
        
        self.noise = 0.1
        self.x_dim = sigma
        self.sigma = sigma
    
    def sim(self, n_samples=1000, rng=np.random):
        
        x = rng.rand(n_samples, self.sigma)
        samples = x
        
        return samples
        
    def density_fun(self, xs):
        
        xs = np.asarray(xs)
        dens = np.ones_like(xs)
            
        return dens
    
    def mcmc_entropy(self, n_samples=1000):
        
        return 0
    

class gamma_exponential:
    '''
    "Entropy Expressions for Multivariate Continuous Distributions", Georges A. Darbellay 
    and Igor Vajda, IEEE TRANSACTIONS ON INFORMATION THEORY, 2000
    '''
    
    def __init__(self, sigma):
        '''
        Density function: Gamma(sigma, 1)*Gamma(1, 1/x)
        '''
        
        self.x_dim = 2
        self.sigma = sigma
    
    def sim(self, n_samples=1000, rng=np.random):
        
        x = rng.gamma(shape=self.sigma, scale=1, size=n_samples)
        y = rng.gamma(shape=1, scale=1/x, size=n_samples)
        
        samples = np.concatenate((x[:,np.newaxis],y[:,np.newaxis]), axis=1)
        
        return samples
    
    def plot_scatter(self, samples=None, n_samples=1000):
        
        if samples is None:
            samples = self.sim(n_samples)
        
        fig, ax = plt.subplots(1,1)
        ax.scatter(samples[:,0], samples[:,1] , c='b', s = 0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        return fig
        
    def density_fun(self, xs):
        from scipy.special import gamma
        xs = np.asarray(xs)
        if xs.ndim == 1:
            xs = xs[np.newaxis,:]
        
        #dens = stats.gamma.pdf(xs[:,0], a=self.sigma, scale=1)*stats.gamma.pdf(xs[:,1], a=1/xs[:,0], scale=1)
        dens = xs[:,0]**self.sigma*np.exp(-xs[:,0]-xs[:,0]*xs[:,1])/gamma(self.sigma)
        
        return dens
    
    def mcmc_entropy(self, n_samples=1000):
        
        return -np.mean(np.log(self.density_fun(self.sim(n_samples))))


class dirichlet:

    def __init__(self, sigma):
        
        sigma = np.asarray(sigma)
        self.x_dim = sigma.shape[0]
        self.sigma = sigma
        self.rv = stats.dirichlet(alpha=sigma)
    
    def sim(self, n_samples=1000, rng=np.random):

        samples = self.rv.rvs(size=n_samples)
        
        return samples
    
    def plot_scatter(self, samples=None, n_samples=1000):
        
        import seaborn as sns
        import pandas as pd
        if samples is None:
            samples = self.sim(n_samples=n_samples)
        g = sns.PairGrid(pd.DataFrame(samples))
        g = g.map_diag(sns.kdeplot, shade=True, color='b', lw=3, legend=False)
        g = g.map_upper(plt.scatter)
        g = g.map_lower(sns.kdeplot)
        
    def density_fun(self, xs):
        xs = np.asarray(xs)
        if xs.ndim == 1:
            return self.density_fun(xs[np.newaxis,:])
        dens = np.empty(xs.shape[0])
        for i in range(xs.shape[0]):
            dens[i] = self.rv.pdf(x=xs[i])
        
        return dens
    
    def mcmc_entropy(self, n_samples=1000):
        
        return self.rv.entropy()
    

class Hybrid_Rosenbrock:
    '''
    Pagani, Filippo, Martin Wiegand, and Saralees Nadarajah. 
    "An n-dimensional Rosenbrock Distribution for MCMC Testing." 
    arXiv preprint arXiv:1903.09556 (2019).
    '''
    def __init__(self, mu, n_1, n_2, a, b):
        
        b = np.asarray(b)
        if b.shape == ():
            b = b*np.ones((n_2, n_1-1))
        self.x_dim = (n_1-1)*n_2+1
        self.mu = mu
        self.n_1 = n_1
        self.n_2 = n_2
        self.a = a
        self.b = b
        
    def sim(self, n_samples=1000, rng=np.random):
        
        samples = self.mu + np.sqrt(0.5/self.a)*rng.randn(n_samples, 1)
            
        for j in range(self.n_2):   
                
            for i in range(self.n_1-1):
                
                if i == 0:
                    new_sample = samples[:, 0][:, np.newaxis]**2 + np.sqrt(0.5/self.b[j,i])*rng.randn(n_samples, 1) 
                else:
                    new_sample = samples[:,-1][:, np.newaxis]**2 + np.sqrt(0.5/self.b[j,i])*rng.randn(n_samples, 1)
                
                samples = np.concatenate((samples, new_sample), axis=1)
                
        return samples
    
    def plot_scatter(self, samples=None, n_samples=1000):
        
        if samples is None:
            samples = self.sim(n_samples=n_samples)
        
        plot_hist_marginals(samples)
        
    def density_fun(self, xs):
        
        xs = np.asarray(xs)
        if xs.ndim == 1:
            return self.density_fun(xs[np.newaxis,:])
        
        dens = stats.norm.pdf(xs[:,0]-self.mu, scale=np.sqrt(0.5/self.a))
        
        for j in range(self.n_2):
            
            for i in range(self.n_1-1):

                if i == 0:  
                    dens = dens*stats.norm.pdf(xs[:, j*(self.n_1-1)+i+1] - xs[:, 0]**2, scale=np.sqrt(0.5/self.b[j,i]))
                else:
                    dens = dens*stats.norm.pdf(xs[:, j*(self.n_1-1)+i+1] - xs[:, j*(self.n_1-1)+i]**2, scale=np.sqrt(0.5/self.b[j,i]))
        
        return dens
    
    def mcmc_entropy(self, n_samples=1000):
        
        return -np.mean(np.log(self.density_fun(self.sim(n_samples))))