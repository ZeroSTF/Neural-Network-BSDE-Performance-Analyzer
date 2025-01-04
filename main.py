import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from tqdm import tqdm
from scipy.stats import norm

class PerformanceAnalyzer:
    def __init__(self):
        self.dimensions = [2, 5, 10, 20, 50, 100]
        self.methods = ['BSDE-DNN', 'Longstaff-Schwartz', 'Finite Difference']
        self.base_params = {
            'S0': 100,
            'K': 100,
            'r': 0.05,
            'sigma': 0.2,
            'T': 1.0,
            'n_steps': 50,
            'n_paths': 10000
        }

    def black_scholes_put(self, S, K, T, r, sigma):
        """Reference Black-Scholes price for European put"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    def longstaff_schwartz(self, paths, K, r, dt):
        """Longstaff-Schwartz method implementation"""
        n_paths, n_steps = paths.shape
        V = np.maximum(K - paths[:, -1], 0)  # Terminal payoff
        
        for t in range(n_steps-2, -1, -1):
            continuation = np.zeros(n_paths)
            exercise = np.maximum(K - paths[:, t], 0)
            
            # Only consider in-the-money paths
            itm = exercise > 0
            if sum(itm) > 0:
                X = paths[itm, t]
                Y = V[itm] * np.exp(-r*dt)
                
                # Polynomial regression
                degree = 2
                A = np.vstack([X**i for i in range(degree+1)]).T
                beta = np.linalg.lstsq(A, Y, rcond=None)[0]
                
                # Update values
                continuation[itm] = sum(beta[i] * paths[itm, t]**i 
                                     for i in range(degree+1))
                V[itm] = np.where(exercise[itm] > continuation[itm],
                                exercise[itm],
                                V[itm] * np.exp(-r*dt))
        
        return V[0]

    class BSDE_DNN(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(dim+1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            return self.network(x)

    def test_precision(self):
        """Test precision across different methods and dimensions"""
        results = {method: [] for method in self.methods}
        computation_times = {method: [] for method in self.methods}
        
        for dim in tqdm(self.dimensions, desc="Testing dimensions"):
            params = self.base_params.copy()
            
            # Generate paths
            paths = np.zeros((params['n_paths'], params['n_steps']+1, dim))
            dt = params['T']/params['n_steps']
            
            for d in range(dim):
                dW = np.random.normal(0, np.sqrt(dt), 
                                    (params['n_paths'], params['n_steps']))
                paths[:,:,d] = self._generate_paths(dW, params)
            
            # Test each method
            for method in self.methods:
                start_time = time()
                if method == 'BSDE-DNN':
                    price = self._test_bsde_dnn(paths, params)
                elif method == 'Longstaff-Schwartz':
                    price = self.longstaff_schwartz(paths[:,:,0], 
                                                  params['K'], 
                                                  params['r'], dt)
                else:  # Finite Difference (only for 1D)
                    if dim == 2:
                        price = self._finite_difference(params)
                    else:
                        price = np.nan
                
                computation_time = time() - start_time
                results[method].append(price)
                computation_times[method].append(computation_time)
        
        self._plot_results(results, computation_times)

    def _generate_paths(self, dW, params):
        """Generate stock price paths"""
        paths = np.zeros((params['n_paths'], params['n_steps']+1))
        paths[:,0] = params['S0']
        
        for t in range(1, params['n_steps']+1):
            paths[:,t] = paths[:,t-1] * np.exp(
                (params['r'] - 0.5*params['sigma']**2)*params['T']/params['n_steps'] + 
                params['sigma']*dW[:,t-1]
            )
        return paths

    def _test_bsde_dnn(self, paths, params):
        """Test BSDE-DNN method"""
        model = self.BSDE_DNN(paths.shape[2])
        optimizer = optim.Adam(model.parameters())
        
        for epoch in range(50):
            optimizer.zero_grad()
            
            # Forward pass
            inputs = torch.tensor(np.concatenate(
                [paths[:,:-1,:], 
                 np.ones((paths.shape[0], paths.shape[1]-1, 1))], 
                axis=2
            ), dtype=torch.float32)
            
            predicted = model(inputs.reshape(-1, inputs.shape[2]))
            target = torch.tensor(
                np.maximum(params['K'] - paths[:,1:,0], 0),
                dtype=torch.float32
            ).reshape(-1, 1)
            
            loss = nn.MSELoss()(predicted, target)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            initial_input = torch.tensor(
                np.concatenate([paths[:,0,:], np.ones((paths.shape[0], 1))], axis=1),
                dtype=torch.float32
            )
            price = model(initial_input).mean().item()
        
        return price

    def _finite_difference(self, params):
        """Implement finite difference method for 1D"""
        # Simplified implementation for comparison
        return self.black_scholes_put(
            params['S0'], params['K'], params['T'], 
            params['r'], params['sigma']
        )

    def _plot_results(self, results, computation_times):
        """Create visualization plots for the report"""
        # 1. Price comparison across dimensions
        plt.figure(figsize=(12, 6))
        for method in self.methods:
            plt.plot(self.dimensions, results[method], 
                    marker='o', label=method)
        plt.xlabel('Dimension')
        plt.ylabel('Option Price')
        plt.title('Prix des Options par Méthode et Dimension')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 2. Computation time comparison
        plt.figure(figsize=(12, 6))
        for method in self.methods:
            plt.plot(self.dimensions, computation_times[method], 
                    marker='o', label=method)
        plt.xlabel('Dimension')
        plt.ylabel('Temps de Calcul (s)')
        plt.title('Temps de Calcul par Méthode et Dimension')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 3. Error analysis (using BS as reference for 1D)
        plt.figure(figsize=(12, 6))
        reference = self.black_scholes_put(
            self.base_params['S0'], 
            self.base_params['K'],
            self.base_params['T'],
            self.base_params['r'],
            self.base_params['sigma']
        )
        
        for method in self.methods:
            errors = [abs(price - reference)/reference * 100 
                     if not np.isnan(price) else np.nan 
                     for price in results[method]]
            plt.plot(self.dimensions, errors, marker='o', label=method)
        
        plt.xlabel('Dimension')
        plt.ylabel('Erreur Relative (%)')
        plt.title('Analyse de l\'Erreur par Dimension')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    analyzer.test_precision()