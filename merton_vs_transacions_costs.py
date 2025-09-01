import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketParams:
    """Paramètres du marché Black-Scholes"""
    mu: float = 0.1      # drift du sous-jacent
    sigma: float = 0.2   # volatilité
    r: float = 0.05      # taux sans risque
    T: float = 1.0       # horizon temporel
    dt: float = 0.01     # pas de temps

@dataclass 
class UtilityParams:
    """Paramètres d'utilité"""
    gamma: float = 2.0   # coefficient CRRA (si gamma=1 -> log utility)
    W0: float = 100.0    # richesse initiale

@dataclass
class TransactionCostParams:
    """Paramètres des coûts de transaction"""
    lambda_cost: float = 0.01  # coût proportionnel

class MertonStrategy:
    """Stratégie de Merton sans friction"""
    
    def __init__(self, market: MarketParams, utility: UtilityParams):
        self.market = market
        self.utility = utility
        
        # Calcul de la proportion optimale de Merton
        if utility.gamma == 1.0:  # Log utility
            self.theta_merton = (market.mu - market.r) / (market.sigma**2)
        else:  # Power utility - FORMULE CORRIGÉE
            self.theta_merton = (market.mu - market.r) / (utility.gamma * market.sigma**2)
    
    def simulate_path(self, n_steps: int, random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Simule un chemin de la stratégie de Merton
        Returns: (wealth_path, stock_proportion_path, nb_rebalances)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        dt = self.market.T / n_steps
        times = np.linspace(0, self.market.T, n_steps + 1)
        
        # Génération du mouvement brownien
        dW = np.sqrt(dt) * np.random.randn(n_steps)
        
        # Prix de l'actif risqué
        S = np.zeros(n_steps + 1)
        S[0] = 100.0  # Prix initial normalisé
        
        for i in range(n_steps):
            S[i+1] = S[i] * np.exp((self.market.mu - 0.5 * self.market.sigma**2) * dt + 
                                  self.market.sigma * dW[i])
        
        # Richesse (stratégie de Merton = proportion constante)
        W = np.zeros(n_steps + 1)
        W[0] = self.utility.W0
        
        # Dans Merton, la proportion est constante donc pas de rééquilibrage nécessaire
        # La richesse suit une géométrique brownienne
        for i in range(n_steps):
            portfolio_return = (self.theta_merton * (S[i+1]/S[i] - np.exp(self.market.r * dt)) + 
                              np.exp(self.market.r * dt) - 1)
            W[i+1] = W[i] * (1 + portfolio_return)
        
        # Proportion d'actions (constante dans Merton)
        stock_prop = np.full(n_steps + 1, self.theta_merton)
        
        # Nombre de rééquilibrages = 0 (théoriquement continu mais pas de coûts)
        nb_rebalances = 0
        
        return W, stock_prop, nb_rebalances

class TransactionCostStrategy:
    """Stratégie avec coûts de transaction et politique de bande"""
    
    def __init__(self, market: MarketParams, utility: UtilityParams, tc: TransactionCostParams):
        self.market = market
        self.utility = utility
        self.tc = tc
        
        # Calcul de la proportion de Merton (cible)
        if utility.gamma == 1.0:
            self.theta_merton = (market.mu - market.r) / (market.sigma**2)
        else:
            self.theta_merton = (market.mu - market.r) / (utility.gamma * market.sigma**2)
            
        # Calcul des bornes de la bande optimale (approximation)
        self._compute_optimal_bands()
    
    def _compute_optimal_bands(self):
        """
        Calcule les bornes optimales de la politique de bande
        Basé sur l'approximation asymptotique pour petits coûts de transaction
        """
        # Approximation de Shreve-Soner pour les bornes optimales
        # Pour gamma != 1 (power utility)
        if self.utility.gamma != 1.0:
            gamma = self.utility.gamma
            sigma = self.market.sigma
            lambda_cost = self.tc.lambda_cost
            
            # Approximation asymptotique pour petits lambda
            # Les bornes sont de l'ordre de lambda^(1/3)
            band_width = 2.0 * (3 * lambda_cost / (4 * gamma * sigma**2))**(1/3)
            
            self.lower_bound = max(0, self.theta_merton - band_width)
            self.upper_bound = min(1, self.theta_merton + band_width)
        else:
            # Pour log utility (gamma = 1)
            sigma = self.market.sigma
            lambda_cost = self.tc.lambda_cost
            band_width = 2.0 * (3 * lambda_cost / (4 * sigma**2))**(1/3)
            
            self.lower_bound = max(0, self.theta_merton - band_width)
            self.upper_bound = min(1, self.theta_merton + band_width)
    
    def simulate_path(self, n_steps: int, random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Simule un chemin avec politique de bande
        Returns: (wealth_path, stock_proportion_path, nb_rebalances)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        dt = self.market.T / n_steps
        
        # Génération du mouvement brownien
        dW = np.sqrt(dt) * np.random.randn(n_steps)
        
        # Prix de l'actif risqué
        S = np.zeros(n_steps + 1)
        S[0] = 100.0
        
        for i in range(n_steps):
            S[i+1] = S[i] * np.exp((self.market.mu - 0.5 * self.market.sigma**2) * dt + 
                                  self.market.sigma * dW[i])
        
        # Initialisation du portefeuille
        W = np.zeros(n_steps + 1)
        W[0] = self.utility.W0
        
        # Proportion initiale = Merton
        theta = np.zeros(n_steps + 1)
        theta[0] = self.theta_merton
        
        # Montants en actions et en cash
        stock_value = theta[0] * W[0]
        cash_value = W[0] - stock_value
        stock_shares = stock_value / S[0]
        
        nb_rebalances = 0
        
        for i in range(n_steps):
            # Évolution naturelle des positions
            stock_value = stock_shares * S[i+1]
            # Cash reste constant (taux r=0 assumé dans le modèle normalisé)
            total_wealth = stock_value + cash_value
            W[i+1] = total_wealth
            
            if total_wealth > 0:
                current_theta = stock_value / total_wealth
                theta[i+1] = current_theta
                
                # Vérification si rééquilibrage nécessaire
                if current_theta < self.lower_bound or current_theta > self.upper_bound:
                    # Rééquilibrage vers la cible Merton
                    target_stock_value = self.theta_merton * total_wealth
                    
                    if current_theta < self.lower_bound:
                        # Acheter des actions
                        amount_to_buy = target_stock_value - stock_value
                        cost = self.tc.lambda_cost * amount_to_buy
                        
                        # Vérifier si on a assez de cash
                        if cash_value >= amount_to_buy + cost:
                            stock_value = target_stock_value
                            cash_value -= (amount_to_buy + cost)
                            stock_shares = stock_value / S[i+1]
                            nb_rebalances += 1
                            
                            # Recalculer la richesse après coûts
                            W[i+1] = stock_value + cash_value
                            theta[i+1] = self.theta_merton if W[i+1] > 0 else 0
                    
                    elif current_theta > self.upper_bound:
                        # Vendre des actions
                        amount_to_sell = stock_value - target_stock_value
                        cost = self.tc.lambda_cost * amount_to_sell
                        
                        stock_value = target_stock_value
                        cash_value += (amount_to_sell - cost)
                        stock_shares = stock_value / S[i+1] if S[i+1] > 0 else 0
                        nb_rebalances += 1
                        
                        # Recalculer la richesse après coûts
                        W[i+1] = stock_value + cash_value
                        theta[i+1] = self.theta_merton if W[i+1] > 0 else 0
            else:
                theta[i+1] = 0
        
        return W, theta, nb_rebalances

def compute_utility(W_T: float, gamma: float) -> float:
    """Calcule l'utilité finale"""
    if gamma == 1.0:  # Log utility
        return np.log(max(W_T, 1e-10))
    else:  # Power utility
        if W_T <= 0:
            return -np.inf
        return (W_T**(1-gamma) - 1) / (1-gamma)

class SimulationEngine:
    """Moteur de simulation principal"""
    
    def __init__(self, market: MarketParams, utility: UtilityParams, tc: TransactionCostParams):
        self.market = market
        self.utility = utility
        self.tc = tc
        
        self.merton_strategy = MertonStrategy(market, utility)
        self.tc_strategy = TransactionCostStrategy(market, utility, tc)
    
    def run_monte_carlo(self, n_simulations: int = 1000, n_steps: int = 100) -> dict:
        """
        Lance une simulation Monte Carlo complète
        """
        print(f"Lancement de {n_simulations} simulations...")
        print(f"Paramètres du marché: μ={self.market.mu}, σ={self.market.sigma}, r={self.market.r}")
        print(f"Utilité: γ={self.utility.gamma}, W₀={self.utility.W0}")
        print(f"Coût de transaction: λ={self.tc.lambda_cost}")
        print(f"Proportion de Merton: θ*={self.merton_strategy.theta_merton:.4f}")
        print(f"Bornes de la bande: [{self.tc_strategy.lower_bound:.4f}, {self.tc_strategy.upper_bound:.4f}]")
        print("-" * 60)
        
        # Stockage des résultats
        results = {
            'merton': {'W_T': [], 'utilities': [], 'nb_rebalances': []},
            'transaction_costs': {'W_T': [], 'utilities': [], 'nb_rebalances': []}
        }
        
        # Simulations avec mêmes graines aléatoires pour comparaison équitable
        for i in range(n_simulations):
            if (i + 1) % 200 == 0:
                print(f"Simulation {i+1}/{n_simulations}")
            
            # Stratégie de Merton (frictionless)
            W_merton, _, nb_rebal_merton = self.merton_strategy.simulate_path(n_steps, random_seed=i)
            utility_merton = compute_utility(W_merton[-1], self.utility.gamma)
            
            results['merton']['W_T'].append(W_merton[-1])
            results['merton']['utilities'].append(utility_merton)
            results['merton']['nb_rebalances'].append(nb_rebal_merton)
            
            # Stratégie avec coûts de transaction
            W_tc, _, nb_rebal_tc = self.tc_strategy.simulate_path(n_steps, random_seed=i)
            utility_tc = compute_utility(W_tc[-1], self.utility.gamma)
            
            results['transaction_costs']['W_T'].append(W_tc[-1])
            results['transaction_costs']['utilities'].append(utility_tc)
            results['transaction_costs']['nb_rebalances'].append(nb_rebal_tc)
        
        return self._analyze_results(results)
    
    def _analyze_results(self, results: dict) -> dict:
        """Analyse les résultats de la simulation"""
        
        analysis = {}
        
        for strategy_name, data in results.items():
            W_T_array = np.array(data['W_T'])
            utilities_array = np.array(data['utilities'])
            nb_rebal_array = np.array(data['nb_rebalances'])
            
            # Filtrer les valeurs infinies pour les statistiques
            finite_utilities = utilities_array[np.isfinite(utilities_array)]
            
            analysis[strategy_name] = {
                'E[W_T]': np.mean(W_T_array),
                'Var[W_T]': np.var(W_T_array),
                'Std[W_T]': np.std(W_T_array),
                'E[U(W_T)]': np.mean(finite_utilities) if len(finite_utilities) > 0 else -np.inf,
                'E[nb_rebalances]': np.mean(nb_rebal_array),
                'Median[W_T]': np.median(W_T_array),
                'Q25[W_T]': np.percentile(W_T_array, 25),
                'Q75[W_T]': np.percentile(W_T_array, 75),
                'Min[W_T]': np.min(W_T_array),
                'Max[W_T]': np.max(W_T_array),
                'Prob[W_T < W0]': np.mean(W_T_array < self.utility.W0),
                'raw_data': data
            }
        
        return analysis
    
    def print_results(self, analysis: dict):
        """Affiche les résultats de manière formatée"""
        
        print("\n" + "="*80)
        print("RÉSULTATS DE LA SIMULATION")
        print("="*80)
        
        strategies = ['merton', 'transaction_costs']
        strategy_labels = ['Stratégie Frictionless (Merton)', 'Stratégie avec Coûts de Transaction']
        
        # Tableau comparatif
        print(f"\n{'Métrique':<25} {'Merton':<20} {'Avec Coûts':<20} {'Différence':<15}")
        print("-" * 80)
        
        metrics = [
            ('E[W_T]', 'E[W_T]'),
            ('Var[W_T]', 'Var[W_T]'),  
            ('E[U(W_T)]', 'E[U(W_T)]'),
            ('E[nb_rebalances]', 'E[nb_rebalances]'),
            ('Median[W_T]', 'Median[W_T]'),
            ('Std[W_T]', 'Std[W_T]')
        ]
        
        for label, key in metrics:
            merton_val = analysis['merton'][key]
            tc_val = analysis['transaction_costs'][key]
            
            if key == 'E[nb_rebalances]':
                diff = f"{tc_val:.1f} - {merton_val:.1f}"
                print(f"{label:<25} {merton_val:<20.1f} {tc_val:<20.1f} {diff:<15}")
            elif np.isfinite(merton_val) and np.isfinite(tc_val):
                diff_abs = tc_val - merton_val
                diff_pct = (diff_abs / merton_val * 100) if merton_val != 0 else 0
                print(f"{label:<25} {merton_val:<20.2f} {tc_val:<20.2f} {diff_pct:<14.1f}%")
            else:
                print(f"{label:<25} {merton_val:<20.2f} {tc_val:<20.2f} {'N/A':<15}")
        
        print("\n" + "="*80)
        
        # Statistiques détaillées pour chaque stratégie
        for i, (strategy, label) in enumerate(zip(strategies, strategy_labels)):
            data = analysis[strategy]
            print(f"\n{label}:")
            print(f"  Espérance de richesse finale    : {data['E[W_T]']:.2f}")
            print(f"  Variance de richesse finale     : {data['Var[W_T]']:.2f}")
            print(f"  Écart-type de richesse finale   : {data['Std[W_T]']:.2f}")
            print(f"  Espérance d'utilité finale      : {data['E[U(W_T)]']:.4f}")
            print(f"  Nombre moyen de rééquilibrages  : {data['E[nb_rebalances]']:.1f}")
            print(f"  Médiane de richesse finale      : {data['Median[W_T]']:.2f}")
            print(f"  Quartiles [Q25, Q75]            : [{data['Q25[W_T]']:.2f}, {data['Q75[W_T]']:.2f}]")
            print(f"  Richesse [Min, Max]             : [{data['Min[W_T]']:.2f}, {data['Max[W_T]']:.2f}]")
            print(f"  Probabilité de perte            : {data['Prob[W_T < W0]']:.1%}")
    
    def plot_sample_paths(self, n_paths: int = 5, n_steps: int = 100):
        """Trace quelques chemins d'exemple"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        times = np.linspace(0, self.market.T, n_steps + 1)
        
        for i in range(n_paths):
            # Merton strategy
            W_merton, theta_merton, _ = self.merton_strategy.simulate_path(n_steps, random_seed=i)
            
            # Transaction costs strategy  
            W_tc, theta_tc, _ = self.tc_strategy.simulate_path(n_steps, random_seed=i)
            
            alpha = 0.7 if n_paths > 1 else 1.0
            ax1.plot(times, W_merton, alpha=alpha, label=f'Path {i+1}' if i == 0 else "")
            ax2.plot(times, W_tc, alpha=alpha, label=f'Path {i+1}' if i == 0 else "")
            ax3.plot(times, theta_merton, alpha=alpha, label=f'Path {i+1}' if i == 0 else "")
            ax4.plot(times, theta_tc, alpha=alpha, label=f'Path {i+1}' if i == 0 else "")
        
        # Configuration des graphiques
        ax1.set_title('Évolution de la Richesse - Merton (Frictionless)')
        ax1.set_xlabel('Temps')
        ax1.set_ylabel('Richesse W(t)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.utility.W0, color='red', linestyle='--', alpha=0.5, label='W₀')
        
        ax2.set_title('Évolution de la Richesse - Avec Coûts de Transaction')
        ax2.set_xlabel('Temps')
        ax2.set_ylabel('Richesse W(t)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=self.utility.W0, color='red', linestyle='--', alpha=0.5, label='W₀')
        
        ax3.set_title('Proportion en Actions - Merton')
        ax3.set_xlabel('Temps')
        ax3.set_ylabel('Proportion θ(t)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=self.merton_strategy.theta_merton, color='red', linestyle='--', alpha=0.7, label='θ* Merton')
        
        ax4.set_title('Proportion en Actions - Avec Coûts de Transaction')
        ax4.set_xlabel('Temps')
        ax4.set_ylabel('Proportion θ(t)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=self.tc_strategy.theta_merton, color='red', linestyle='--', alpha=0.7, label='θ* cible')
        ax4.axhline(y=self.tc_strategy.lower_bound, color='orange', linestyle=':', alpha=0.7, label='Bornes')
        ax4.axhline(y=self.tc_strategy.upper_bound, color='orange', linestyle=':', alpha=0.7)
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_distributions(self, analysis: dict):
        """Trace les distributions de richesse finale et d'utilité"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution de richesse finale
        W_merton = analysis['merton']['raw_data']['W_T']
        W_tc = analysis['transaction_costs']['raw_data']['W_T']
        
        ax1.hist(W_merton, bins=50, alpha=0.7, density=True, label='Merton', color='blue')
        ax1.hist(W_tc, bins=50, alpha=0.7, density=True, label='Avec coûts', color='red')
        ax1.set_xlabel('Richesse finale W_T')
        ax1.set_ylabel('Densité')
        ax1.set_title('Distribution de la Richesse Finale')
        ax1.axvline(x=self.utility.W0, color='black', linestyle='--', alpha=0.5, label='W₀')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribution d'utilité
        U_merton = analysis['merton']['raw_data']['utilities']
        U_tc = analysis['transaction_costs']['raw_data']['utilities']
        
        # Filtrer les valeurs finies
        U_merton_finite = [u for u in U_merton if np.isfinite(u)]
        U_tc_finite = [u for u in U_tc if np.isfinite(u)]
        
        if U_merton_finite and U_tc_finite:
            ax2.hist(U_merton_finite, bins=50, alpha=0.7, density=True, label='Merton', color='blue')
            ax2.hist(U_tc_finite, bins=50, alpha=0.7, density=True, label='Avec coûts', color='red')
        
        ax2.set_xlabel('Utilité finale U(W_T)')
        ax2.set_ylabel('Densité')
        ax2.set_title('Distribution de l\'Utilité Finale')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Nombre de rééquilibrages
        nb_rebal_tc = analysis['transaction_costs']['raw_data']['nb_rebalances']
        
        ax3.hist(nb_rebal_tc, bins=range(max(nb_rebal_tc) + 2), alpha=0.7, color='green')
        ax3.set_xlabel('Nombre de rééquilibrages')
        ax3.set_ylabel('Fréquence')
        ax3.set_title('Distribution du Nombre de Rééquilibrages\n(Stratégie avec Coûts)')
        ax3.grid(True, alpha=0.3)
        
        # Comparaison directe richesse finale
        ax4.scatter(W_merton, W_tc, alpha=0.3, s=10)
        min_w = min(min(W_merton), min(W_tc))
        max_w = max(max(W_merton), max(W_tc))
        ax4.plot([min_w, max_w], [min_w, max_w], 'r--', alpha=0.7, label='Diagonale')
        ax4.set_xlabel('Richesse Merton')
        ax4.set_ylabel('Richesse avec Coûts')
        ax4.set_title('Comparaison Richesse Finale\n(chaque point = 1 simulation)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Fonction principale pour exécuter la simulation
def run_complete_simulation():
    """
    Fonction principale pour exécuter une simulation complète
    """
    print("="*80)
    print("SIMULATION COMPLÈTE: MERTON vs COÛTS DE TRANSACTION")
    print("="*80)
    
    # Paramètres du marché
    market = MarketParams(
        mu=0.10,      # 10% de drift annuel
        sigma=0.20,   # 20% de volatilité annuelle  
        r=0.03,       # 3% taux sans risque
        T=1.0,        # 1 an
        dt=0.01       # pas de temps
    )
    
    # Paramètres d'utilité CRRA
    utility = UtilityParams(
        gamma=2.0,    # Aversion au risque modérée
        W0=100.0      # 100€ initial
    )
    
    # Coûts de transaction
    tc = TransactionCostParams(
        lambda_cost=0.005  # 0.5% de coût proportionnel
    )
    
    # Création du moteur de simulation
    sim = SimulationEngine(market, utility, tc)
    
    # Simulation Monte Carlo
    analysis = sim.run_monte_carlo(n_simulations=2000, n_steps=252)  # 252 jours de trading
    
    # Affichage des résultats
    sim.print_results(analysis)
    
    # Graphiques
    print("\nGénération des graphiques...")
    sim.plot_sample_paths(n_paths=3, n_steps=252)
    sim.plot_distributions(analysis)
    
    print("\nSimulation terminée!")
    return sim, analysis

if __name__ == "__main__":
    # Exécuter la simulation
    simulation_engine, results = run_complete_simulation()