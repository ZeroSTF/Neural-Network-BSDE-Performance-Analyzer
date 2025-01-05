import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from tqdm import tqdm
from scipy.stats import norm
from typing import Dict, List, Tuple

class BSDE_DNN(nn.Module):
    def __init__(self, dim: int):
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
        
    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes price for a European put option"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
        """Calculate option Greeks"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Delta
        delta = -norm.cdf(-d1)
        
        # Gamma
        gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
        
        # Theta
        theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)
        
        # Vega
        vega = S*np.sqrt(T)*norm.pdf(d1)
        
        # Rho
        rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    def longstaff_schwartz(self, paths: np.ndarray, K: float, r: float, dt: float) -> float:
        """Implement Longstaff-Schwartz method"""
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
        
        return V.mean()

    def _finite_difference(self, params: Dict) -> float:
        """Implement finite difference method"""
        # Using Black-Scholes as a simplified implementation for comparison
        return self.black_scholes_put(
            params['S0'],
            params['K'],
            params['T'],
            params['r'],
            params['sigma']
        )

    def _test_bsde_dnn(self, paths: np.ndarray, params: Dict) -> float:
        """Test BSDE-DNN method"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BSDE_DNN(paths.shape[2]).to(device)
        optimizer = optim.Adam(model.parameters())
        
        for epoch in range(50):
            optimizer.zero_grad()
            
            # Forward pass
            inputs = torch.tensor(np.concatenate(
                [paths[:,:-1,:], 
                 np.ones((paths.shape[0], paths.shape[1]-1, 1))], 
                axis=2
            ), dtype=torch.float32).to(device)
            
            predicted = model(inputs.reshape(-1, inputs.shape[2]))
            target = torch.tensor(
                np.maximum(params['K'] - paths[:,1:,0], 0),
                dtype=torch.float32
            ).reshape(-1, 1).to(device)
            
            loss = nn.MSELoss()(predicted, target)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            initial_input = torch.tensor(
                np.concatenate([paths[:,0,:], np.ones((paths.shape[0], 1))], axis=1),
                dtype=torch.float32
            ).to(device)
            price = model(initial_input).mean().item()
        
        return price

    def analyze_stability(self, n_trials: int = 10) -> Dict[str, Dict[str, List[float]]]:
        """Analyze numerical stability across multiple trials"""
        stability_results = {method: {'prices': [], 'std_dev': []} for method in self.methods}
        
        for dim in tqdm(self.dimensions, desc="Analyzing stability"):
            for method in self.methods:
                prices = []
                for _ in range(n_trials):
                    paths = self._generate_multi_dim_paths(dim)
                    
                    if method == 'BSDE-DNN':
                        price = self._test_bsde_dnn(paths, self.base_params)
                    elif method == 'Longstaff-Schwartz':
                        price = self.longstaff_schwartz(
                            paths[:,:,0], 
                            self.base_params['K'], 
                            self.base_params['r'], 
                            self.base_params['T']/self.base_params['n_steps']
                        )
                    else:  # Finite Difference
                        if dim == 2:
                            price = self._finite_difference(self.base_params)
                        else:
                            price = np.nan
                            
                    prices.append(price)
                
                stability_results[method]['prices'].append(np.mean(prices))
                stability_results[method]['std_dev'].append(np.std(prices))
        
        return stability_results

    def _generate_multi_dim_paths(self, dim: int) -> np.ndarray:
        """Generate multi-dimensional correlated paths"""
        params = self.base_params
        paths = np.zeros((params['n_paths'], params['n_steps']+1, dim))
        dt = params['T']/params['n_steps']
        
        # Generate correlated Brownian motions
        correlation = np.eye(dim) * 0.8 + np.ones((dim, dim)) * 0.2
        cholesky = np.linalg.cholesky(correlation)
        
        for t in range(1, params['n_steps']+1):
            dW = np.random.normal(0, np.sqrt(dt), (params['n_paths'], dim))
            dW = np.dot(dW, cholesky.T)
            
            paths[:,t,:] = paths[:,t-1,:] * np.exp(
                (params['r'] - 0.5*params['sigma']**2)*dt + 
                params['sigma']*dW
            )
        
        paths[:,0,:] = params['S0']
        return paths

    def comprehensive_analysis(self):
        """Perform comprehensive analysis including precision, Greeks, and stability"""
        # show available styles in the console

        print(plt.style.available)
        # 1. Basic precision and timing analysis
        results, computation_times = self.test_precision()
        
        # 2. Greeks analysis
        greeks = self.calculate_greeks(
            self.base_params['S0'],
            self.base_params['K'],
            self.base_params['T'],
            self.base_params['r'],
            self.base_params['sigma']
        )
        
        # 3. Stability analysis
        stability_results = self.analyze_stability()
        
        # Generate comprehensive plots
        self._plot_comprehensive_results(
            results, computation_times, greeks, stability_results
        )

    def test_precision(self) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Test precision across different methods and dimensions"""
        results = {method: [] for method in self.methods}
        computation_times = {method: [] for method in self.methods}
        
        for dim in tqdm(self.dimensions, desc="Testing dimensions"):
            paths = self._generate_multi_dim_paths(dim)
            
            for method in self.methods:
                start_time = time()
                if method == 'BSDE-DNN':
                    price = self._test_bsde_dnn(paths, self.base_params)
                elif method == 'Longstaff-Schwartz':
                    price = self.longstaff_schwartz(
                        paths[:,:,0],
                        self.base_params['K'],
                        self.base_params['r'],
                        self.base_params['T']/self.base_params['n_steps']
                    )
                else:  # Finite Difference
                    if dim == 2:
                        price = self._finite_difference(self.base_params)
                    else:
                        price = np.nan
                
                computation_time = time() - start_time
                results[method].append(price)
                computation_times[method].append(computation_time)
        
        return results, computation_times

    def _plot_comprehensive_results(
        self, 
        results: Dict[str, List[float]], 
        computation_times: Dict[str, List[float]],
        greeks: Dict[str, float],
        stability_results: Dict[str, Dict[str, List[float]]]
    ):
        """Create comprehensive visualization plots"""
        plt.style.use('seaborn-v0_8')
        
        # 1. Price comparison across dimensions
        self._plot_price_comparison(results)
        
        # 2. Computation time analysis
        self._plot_computation_time(computation_times)
        
        # 3. Greeks visualization
        self._plot_greeks(greeks)
        
        # 4. Stability analysis
        self._plot_stability_analysis(stability_results)
        
        # 5. Error analysis with confidence intervals
        self._plot_error_analysis(results, stability_results)

    def _plot_price_comparison(self, results: Dict[str, List[float]]):
        plt.figure(figsize=(12, 6))
        for method in self.methods:
            plt.plot(self.dimensions, results[method], 
                    marker='o', label=method)
        plt.xlabel('Dimension')
        plt.ylabel('Prix de l\'Option')
        plt.title('Comparaison des Prix par Méthode et Dimension')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _plot_computation_time(self, computation_times: Dict[str, List[float]]):
        plt.figure(figsize=(12, 6))
        for method in self.methods:
            plt.semilogy(self.dimensions, computation_times[method], 
                        marker='o', label=method)
        plt.xlabel('Dimension')
        plt.ylabel('Temps de Calcul (s)')
        plt.title('Analyse de la Scalabilité')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _plot_greeks(self, greeks: Dict[str, float]):
        plt.figure(figsize=(12, 6))
        greek_names = list(greeks.keys())
        greek_values = list(greeks.values())
        
        sns.barplot(x=greek_names, y=greek_values)
        plt.title('Analyse des Greeks')
        plt.ylabel('Valeur')
        plt.xticks(rotation=45)
        plt.show()

    def _plot_stability_analysis(self, stability_results: Dict[str, Dict[str, List[float]]]):
        plt.figure(figsize=(12, 6))
        for method in self.methods:
            mean_prices = stability_results[method]['prices']
            std_dev = stability_results[method]['std_dev']
            
            plt.errorbar(self.dimensions, mean_prices, yerr=std_dev, 
                        label=method, marker='o', capsize=5)
        
        plt.xlabel('Dimension')
        plt.ylabel('Prix Moyen ± Écart-Type')
        plt.title('Analyse de la Stabilité Numérique')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _plot_error_analysis(
        self, 
        results: Dict[str, List[float]], 
        stability_results: Dict[str, Dict[str, List[float]]]
    ):
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
            std_dev = stability_results[method]['std_dev']
            
            plt.errorbar(self.dimensions, errors, yerr=std_dev, 
                        label=method, marker='o', capsize=5)
        
        plt.xlabel('Dimension')
        plt.ylabel('Erreur Relative (%)')
        plt.title('Analyse de l\'Erreur avec Intervalles de Confiance')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    analyzer.comprehensive_analysis()