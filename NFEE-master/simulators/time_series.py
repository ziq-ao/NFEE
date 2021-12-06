import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from util.plot import plot_hist_marginals


class M2:
    
    def __init__(self, theta1, theta2, N):
        
        self.theta1 = theta1
        self.theta2 = theta2
        self.N = N
        
    def sim(self, n_samples=1000, rng=np.random):
        
        xs = np.empty((n_samples, self.N))
        ws = np.empty((n_samples, self.N+2))
        ws[:, :2] = rng.randn(n_samples, 2)
        
        for i in range(n_samples):
            
            for j in range(self.N):
                
                ws[i, j+2] = rng.randn()
                xs[i, j] = ws[i, j+2] + self.theta1*ws[i, j+1] + self.theta2*ws[i, j]
        
        return xs, ws
    

class LV:
    
    def __init__(self, n_sensors):
        
        self.n_sensors = n_sensors
        self.x_dim = 2*n_sensors
        self.tmax = 15
        self.timestep = 0.01
        self.a_interval = [0.5, 5] 
        self.b_interval = [0.5, 5] 
        measure_time = []
        for i in range(n_sensors):
            measure_time.append((i+1)*int(self.tmax/self.timestep/(n_sensors+1)))
        self.measure_time = measure_time
        self.noise_std = 0.1
        
    def sim(self, n_samples=1000, rng=np.random):
        
        xs = np.empty((n_samples, 2*self.n_sensors))

        for i in range(n_samples):
            a = self.a_interval[0]+(self.a_interval[1]-self.a_interval[0])*np.random.rand()
            b = self.b_interval[0]+(self.b_interval[1]-self.b_interval[0])*np.random.rand()
            sim_model = Lotka_Volterra(a,1,b,1,self.tmax,self.timestep)
            sim_model.set_initial_conditions(0.5, 1)
            sim_model.integrate_stochastic()
            xs[i, :self.n_sensors] = sim_model.predator[self.measure_time]+self.noise_std*np.random.randn(self.n_sensors)
            xs[i, self.n_sensors:] = sim_model.prey[self.measure_time]+self.noise_std*np.random.randn(self.n_sensors)
        
        return xs
    
    def sim_joint(self, n_samples=1000, rng=np.random):
        
        xs = np.empty((n_samples, 2 + 2*self.n_sensors))

        for i in range(n_samples):
            a = self.a_interval[0]+(self.a_interval[1]-self.a_interval[0])*np.random.rand()
            b = self.b_interval[0]+(self.b_interval[1]-self.b_interval[0])*np.random.rand()
            sim_model = Lotka_Volterra(a,1,b,1,self.tmax,self.timestep)
            sim_model.set_initial_conditions(0.5, 1)
            sim_model.integrate_stochastic()
            xs[i, 0] = a
            xs[i, 1] = b
            xs[i, 2:self.n_sensors+2] = sim_model.predator[self.measure_time]+self.noise_std*np.random.randn(self.n_sensors)
            xs[i, self.n_sensors+2:] = sim_model.prey[self.measure_time]+self.noise_std*np.random.randn(self.n_sensors)
        
        return xs
    
    
    def plot_scatter(self, samples=None, n_samples=1000):
        
        if samples is None:
            samples = self.sim(n_samples)
        
        fig = plot_hist_marginals(samples)
        
        return fig
    
    def plot_many_sims(self, n_samples=1000):
        
        fig, ax = plt.subplots(1,1)
        
        predator_data = []
        prey_data = []
        for i in range(n_samples):
            a = self.a_interval[0]+(self.a_interval[1]-self.a_interval[0])*np.random.rand()
            b = self.b_interval[0]+(self.b_interval[1]-self.b_interval[0])*np.random.rand()
            sim_model = Lotka_Volterra(a,1,b,1,self.tmax,self.timestep)
            sim_model.set_initial_conditions(0.5, 1)
            sim_model.integrate_stochastic()
            
            predator = sim_model.predator
            prey = sim_model.prey
            ax.plot(sim_model.time, predator+self.noise_std*np.random.randn(predator.shape[0]), '-b', label='Predator',linewidth=0.1, alpha=0.5)
            ax.plot(sim_model.time, prey+self.noise_std*np.random.randn(prey.shape[0]), '-r', label='Prey', linewidth=0.1, alpha=0.5)
            
            predator_data.append(predator+self.noise_std*np.random.randn(predator.shape[0]))
            prey_data.append(prey+self.noise_std*np.random.randn(prey.shape[0]))
            
        np.savetxt('prey',prey_data)
        np.savetxt('predator', predator_data)
        

class LV2:
    
    def __init__(self, n_sensors):
        
        self.n_sensors = n_sensors
        self.x_dim = 2*n_sensors
        self.tmax = 10
        self.timestep = 0.01
        self.a_interval = [0.5, 4] 
        self.b_interval = [0.5, 4] 
        measure_time = []
        for i in range(n_sensors):
            measure_time.append((i+1)*int(self.tmax/self.timestep/(n_sensors+1)))
        self.measure_time = measure_time
        self.noise_std = 0.1
        
    def sim(self, n_samples=1000, rng=np.random):
        
        xs = np.empty((n_samples, 2*self.n_sensors))

        for i in range(n_samples):
            a = self.a_interval[0]+(self.a_interval[1]-self.a_interval[0])*np.random.rand()
            b = self.b_interval[0]+(self.b_interval[1]-self.b_interval[0])*np.random.rand()
            sim_model = Lotka_Volterra(a,1,b,1,self.tmax,self.timestep)
            sim_model.set_initial_conditions(0.5, 1)
            sim_model.integrate()
            xs[i, :self.n_sensors] = sim_model.predator[self.measure_time]+self.noise_std*np.random.randn(self.n_sensors)
            xs[i, self.n_sensors:] = sim_model.prey[self.measure_time]+self.noise_std*np.random.randn(self.n_sensors)
        
        return xs
    
    def sim_forward(self, n_samples=1000, rng=np.random):
        
        xs = np.empty((n_samples, 2*self.n_sensors))

        for i in range(n_samples):
            a = self.a_interval[0]+(self.a_interval[1]-self.a_interval[0])*np.random.rand()
            b = self.b_interval[0]+(self.b_interval[1]-self.b_interval[0])*np.random.rand()
            sim_model = Lotka_Volterra(a,1,b,1,self.tmax,self.timestep)
            sim_model.set_initial_conditions(0.5, 1)
            sim_model.integrate()
            xs[i, :self.n_sensors] = sim_model.predator[self.measure_time]
            xs[i, self.n_sensors:] = sim_model.prey[self.measure_time]
        return xs
    
    def sim_joint(self, n_samples=1000, rng=np.random):
        
        xs = np.empty((n_samples, 2 + 2*self.n_sensors))

        for i in range(n_samples):
            a = self.a_interval[0]+(self.a_interval[1]-self.a_interval[0])*np.random.rand()
            b = self.b_interval[0]+(self.b_interval[1]-self.b_interval[0])*np.random.rand()
            sim_model = Lotka_Volterra(a,1,b,1,self.tmax,self.timestep)
            sim_model.set_initial_conditions(0.5, 1)
            sim_model.integrate()
            xs[i, 0] = a
            xs[i, 1] = b
            xs[i, 2:self.n_sensors+2] = sim_model.predator[self.measure_time]+self.noise_std*np.random.randn(self.n_sensors)
            xs[i, self.n_sensors+2:] = sim_model.prey[self.measure_time]+self.noise_std*np.random.randn(self.n_sensors)
        
        return xs
    
    def mcmc_entropy(self, n_out=1000, n_in=1000):
        
        xs = self.sim(n_samples=n_out)
        
        p_y = np.empty(n_out)
        for i in range(n_out):
            ys = self.sim_forward(n_samples=n_in)
            p_yx = np.prod(stats.norm.pdf(xs[i]-ys, scale=self.noise_std), axis=1)
                
            p_y[i] = np.mean(p_yx)
        
        return -np.mean(np.log(p_y)), np.std(np.log(p_y))/np.sqrt(n_out)
    
    def plot_scatter(self, samples=None, n_samples=1000):
        
        if samples is None:
            samples = self.sim(n_samples)
        
        fig = plot_hist_marginals(samples)
        
        return fig
    
    def plot_many_sims(self, n_samples=1000):
        
        fig, ax = plt.subplots(1,1)
        
        predator_data = []
        prey_data = []
        for i in range(n_samples):
            a = self.a_interval[0]+(self.a_interval[1]-self.a_interval[0])*np.random.rand()
            b = self.b_interval[0]+(self.b_interval[1]-self.b_interval[0])*np.random.rand()
            sim_model = Lotka_Volterra(a,1,b,1,self.tmax,self.timestep)
            sim_model.set_initial_conditions(0.5, 1)
            sim_model.integrate()
            
            predator = sim_model.predator
            prey = sim_model.prey
            ax.plot(sim_model.time, predator+self.noise_std*np.random.randn(predator.shape[0]), '-b', label='Predator',linewidth=0.1, alpha=0.5)
            ax.plot(sim_model.time, prey+self.noise_std*np.random.randn(prey.shape[0]), '-r', label='Prey', linewidth=0.1, alpha=0.5)
            
            predator_data.append(predator+self.noise_std*np.random.randn(predator.shape[0]))
            prey_data.append(prey+self.noise_std*np.random.randn(prey.shape[0]))
            
        np.savetxt('prey',prey_data)
        np.savetxt('predator', predator_data)


class Lotka_Volterra:
    '''Sets up a simple Lotka_Volterra system'''
    
    def __init__(self, pdgrow,pddie,pygrow,pydie,tmax,timestep,prey_capacity=100.0,predator_capacity = 100.0):
        '''Create Lotka-Volterra system'''
        
        self.n = int(tmax/timestep)
        self.dt = timestep
        self.time = np.zeros(self.n)
        self.prey = np.zeros(self.n)
        self.predator = np.zeros(self.n)
        self.preygrow = pygrow
        self.preydie = pydie
        self.predgrow = pdgrow
        self.preddie = pddie
        self.prey_capacity = prey_capacity
        self.predator_capacity = predator_capacity
        
    def set_initial_conditions(self,pdzero,pyzero, tzero=0.0):
        '''set initial conditions'''
        self.prey[0] = pyzero
        self.predator[0] = pdzero
        self.time[0] = tzero
        
    def integrate(self):
        '''integrate vanilla Lotka-Volterra system (simple Euler method)'''
        for i in range(self.n-1):
                        
            self.time[i+1] = self.time[i] + self.dt
            self.predator[i+1] = self.predator[i] + self.dt*self.predator[i]*(self.predgrow*self.prey[i] - self.preddie)
            self.prey[i+1] = self.prey[i] + self.dt*self.prey[i]*(self.preygrow - self.predator[i]*self.preydie)

    
    def integrate_logistic(self):
        '''integrate Lotka-Volterra system assuming logistic growth (simple Euler method)'''
        
        for i in range(self.n-1):
            self.time[i+1] = self.time[i]+self.dt
            self.predator[i+1] = self.predator[i] + self.dt*self.predator[i]*(self.predgrow*self.prey[i]*(1.0 - self.predator[i]/self.predator_capacity) - self.preddie)
            self.prey[i+1] = self.prey[i] + self.dt*self.prey[i]*self.preygrow*(1.0-self.prey[i]/self.prey_capacity) - self.prey[i]*self.predator[i]*self.preydie
            #print self.time[i], self.predator[i],self.prey[i]
    
    def integrate_stochastic(self):
        '''integrate vanilla Lotka-Volterra system with stochastic predator death rate (simple Euler method)'''
        for i in range(self.n-1):
                        
            self.time[i+1] = self.time[i] + self.dt
            self.predator[i+1] = self.predator[i] + self.dt*self.predator[i]*(self.predgrow*self.prey[i] - self.preddie*(1-0.1)*np.random.rand())
            self.prey[i+1] = self.prey[i] + self.dt*self.prey[i]*(self.preygrow*(1-0.1)*np.random.rand() - self.predator[i]*self.preydie)

            
    def plot_vs_time(self, filename='populations_vs_time.png', plot_capacity=False):
        
        '''Plot both populations vs time'''
        predlabel = 'Predator Count (Thousands)'
        preylabel = 'Prey Count (Thousands)'

        predcolor = 'red'
        preycolor = 'blue'
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_xlabel('Time', fontsize=22)
        ax1.set_ylabel(predlabel,fontsize=22, color=predcolor)
        ax1.tick_params('y', colors=predcolor)
        ax2.set_ylabel(preylabel,fontsize=22, color=preycolor)
        ax2.tick_params('y', colors='blue', color=preycolor)
        ax1.plot(self.time, self.predator, label='Predator', color=predcolor, linestyle='dashed')
        ax2.plot(self.time, self.prey, label = 'Prey', color = preycolor)
        if(plot_capacity):
            ax2.axhline(self.prey_capacity, label= 'Prey carrying capacity', color=preycolor, linestyle='dotted')
        #ax2.axhline(self.predator_capacity, label= 'Predator carrying capacity', color=predcolor, linestyle='dashed')
        plt.show()
        fig1.savefig(filename, dpi=300)
        
    def plot_predator_vs_prey(self, filename = 'predator_vs_prey.png'):
        
        '''Plot predators vs prey'''
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        
        predlabel = 'Predator Count (Thousands)'
        preylabel = 'Prey Count (Thousands)'
        
        ax1.set_xlabel(predlabel,fontsize=22)
        ax1.set_ylabel(preylabel,fontsize=22)
        ax1.plot(self.predator,self.prey, color='black')
        plt.show()
        fig1.savefig(filename,dpi=300)
        
    def plot_both_figures(self):
        
        '''Plot both populations vs time & predators vs prey'''
        fig1 = plt.figure()
        
        timelabel = 'Time'
        predlabel = 'Predator Count (Thousands)'
        preylabel = 'Prey Count (Thousands)'
        predcolor = 'red'
        preycolor = 'blue'
        
        ax1 = fig1.add_subplot(211)
        ax2 = ax1.twinx()
        ax1.set_xlabel(timelabel)
        ax1.set_ylabel(predlabel, color=predcolor)
        ax2.set_ylabel(preylabel, color =preycolor)
        ax1.plot(self.time, self.predator, label='Predator', color=predcolor)
        ax2.plot(self.time, self.prey, label = 'Prey', color = preycolor)
        ax1.legend()
        
        ax3 = fig1.add_subplot(212)
        
        ax3.set_xlabel(predlabel)
        ax3.set_ylabel(preylabel)
        ax3.plot(self.predator,self.prey, color = 'black')
        
        plt.show()
        


class PK:
    
    def __init__(self, n_sensors, D=400, sig_prop=0.1, sig_add=np.sqrt(0.1)):
        
        self.D = D
        self.sig_prop = sig_prop
        self.sig_add = sig_add
        
        self.n_sensors = n_sensors
        self.x_dim = n_sensors
        self.tmax = 24.0
        measure_time = np.array(range(1, n_sensors+1))*self.tmax/(n_sensors+1)
        self.measure_time = measure_time
        
    def sim(self, n_samples=1000, rng=np.random):
        
        xs = np.empty((n_samples, self.n_sensors))
        
        for i in range(n_samples):
            ka = np.exp(np.log(1.0) + np.sqrt(0.05)*rng.randn())
            ke = np.exp(np.log(0.1) + np.sqrt(0.05)*rng.randn())
            V = np.exp(np.log(20.0) + np.sqrt(0.05)*rng.randn())  
            xs[i] = self.D/V*ka/(ka-ke)*(np.exp(-ke*self.measure_time)-np.exp(-ka*self.measure_time))*(1+self.sig_prop*rng.randn(self.n_sensors))+self.sig_add*rng.randn(self.n_sensors)
            
        return xs
    
    def plot_scatter(self, samples=None, n_samples=1000):
        
        if samples is None:
            samples = self.sim(n_samples)
            
        lb = np.min(samples, axis=0)
        ub = np.max(samples, axis=0)
        lims = np.array(zip(lb,ub))
        fig = plot_hist_marginals(samples, lims=lims)
        
        return fig
        
    def plot_response(self, n_samples=100, rng=np.random):
        
        t = np.linspace(0.0, self.tmax, 100)
        n = t.shape[0]
        for i in range(n_samples):
            ka = np.exp(np.log(1.0) + np.sqrt(0.05)*rng.randn())
            ke = np.exp(np.log(0.1) + np.sqrt(0.05)*rng.randn())
            V = np.exp(np.log(20.0) + np.sqrt(0.05)*rng.randn())  
            y = self.D/V*ka/(ka-ke)*(np.exp(-ke*t)-np.exp(-ka*t))*(1+self.sig_prop*rng.randn(n))+self.sig_add*rng.randn(n)
            plt.plot(t, y)

        
        
        
       
        
            
            
            
            
            
        
        
