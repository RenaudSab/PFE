"""
markowitz_base.py - Modèle de Markowitz Simple et Efficace
===========================================================

Version simplifiée avec cvxopt pour résoudre les problèmes d'optimisation
- Contraintes de non-négativité automatiques
- Frontière efficiente robuste
- Données de marché réalistes
- Graphiques intégrés
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Essayer d'importer cvxopt, sinon utiliser scipy
try:
    import cvxopt as opt
    from cvxopt import blas, solvers
    solvers.options['show_progress'] = False  # Supprime les messages
    USE_CVXOPT = True
except ImportError:
    print("⚠️  cvxopt non disponible, utilisation de scipy")
    from scipy.optimize import minimize
    USE_CVXOPT = False

# =============================
# DONNÉES DE MARCHÉ RÉALISTES
# =============================

def get_market_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Données de marché réalistes basées sur la thèse
    """
    # Rendements annuels moyens (%)
    expected_returns = np.array([0.10, 0.08, 0.03, 0.06])  # US, EU, Bonds, Commodities
    
    # Volatilités annuelles (%)
    volatilities = np.array([0.20, 0.22, 0.05, 0.25])
    
    # Matrice de corrélation réaliste
    correlations = np.array([
        [1.00, 0.75, 0.20, 0.30],  # US Stocks
        [0.75, 1.00, 0.25, 0.35],  # EU Stocks  
        [0.20, 0.25, 1.00, 0.10],  # Bonds
        [0.30, 0.35, 0.10, 1.00]   # Commodities
    ])
    
    # Construction matrice de covariance
    covariance_matrix = np.outer(volatilities, volatilities) * correlations
    
    asset_names = ['US Stocks', 'EU Stocks', 'Bonds', 'Commodities']
    
    return expected_returns, covariance_matrix, asset_names

def simulate_returns(mu: np.ndarray, Sigma: np.ndarray, n_obs: int = 252) -> np.ndarray:
    """
    Simule des rendements journaliers à partir des paramètres annuels
    """
    np.random.seed(42)  # Pour la reproductibilité
    
    # Conversion annuel -> journalier
    mu_daily = mu / 252
    Sigma_daily = Sigma / 252
    
    # Simulation multivariate normale
    returns = np.random.multivariate_normal(mu_daily, Sigma_daily, n_obs).T
    
    return returns

# =============================
# GÉNÉRATION DE PORTEFEUILLES
# =============================

def rand_weights(n: int) -> np.ndarray:
    """Génère n poids aléatoires qui somment à 1"""
    k = np.random.rand(n)
    return k / sum(k)

def random_portfolio(returns: np.ndarray) -> Tuple[float, float]:
    """Calcule moyenne et écart-type d'un portefeuille aléatoire"""
    n_assets = returns.shape[0]
    weights = rand_weights(n_assets)
    
    portfolio_return = np.mean(returns, axis=1) @ weights
    portfolio_variance = weights.T @ np.cov(returns) @ weights
    portfolio_std = np.sqrt(portfolio_variance)
    
    return portfolio_return, portfolio_std

def generate_random_portfolios(returns: np.ndarray, n_portfolios: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Génère n_portfolios portefeuilles aléatoires"""
    means = np.zeros(n_portfolios)
    stds = np.zeros(n_portfolios)
    
    for i in range(n_portfolios):
        means[i], stds[i] = random_portfolio(returns)
    
    return means, stds

# =============================
# OPTIMISATION MARKOWITZ
# =============================

def optimal_portfolio_cvxopt(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule la frontière efficiente avec cvxopt (méthode robuste)
    """
    n = returns.shape[0]
    
    # Paramètres statistiques
    mu = np.mean(returns, axis=1)
    Sigma = np.cov(returns)
    
    # Gamme de rendements cibles
    N = 50
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    target_returns = np.linspace(mu_min, mu_max * 0.95, N)
    
    # Matrices pour cvxopt
    P = opt.matrix(2.0 * Sigma)  # 2 * Sigma pour la forme quadratique
    q = opt.matrix(0.0, (n, 1))
    
    # Contraintes d'inégalité: -w <= 0 (i.e., w >= 0)
    G = opt.matrix(-np.eye(n))
    h = opt.matrix(0.0, (n, 1))
    
    # Contraintes d'égalité: sum(w) = 1 et mu^T * w = target
    A = opt.matrix(np.vstack([np.ones((1, n)), mu.reshape(1, -1)]))
    
    portfolios = []
    frontier_returns = []
    frontier_risks = []
    
    for target_mu in target_returns:
        b = opt.matrix([1.0, target_mu])
        
        try:
            # Résolution du problème quadratique
            sol = solvers.qp(P, q, G, h, A, b)
            
            if sol['status'] == 'optimal':
                w = np.array(sol['x']).flatten()
                portfolios.append(w)
                
                # Calcul rendement et risque
                port_return = mu.T @ w
                port_risk = np.sqrt(w.T @ Sigma @ w)
                
                frontier_returns.append(port_return)
                frontier_risks.append(port_risk)
                
        except Exception:
            continue
    
    return np.array(portfolios), np.array(frontier_returns), np.array(frontier_risks)

def optimal_portfolio_scipy(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Version fallback avec scipy si cvxopt non disponible
    """
    from scipy.optimize import minimize
    
    n = returns.shape[0]
    mu = np.mean(returns, axis=1)
    Sigma = np.cov(returns)
    
    # Gamme de rendements cibles
    N = 50
    target_returns = np.linspace(np.min(mu), np.max(mu) * 0.95, N)
    
    portfolios = []
    frontier_returns = []
    frontier_risks = []
    
    for target_mu in target_returns:
        # Fonction objectif: minimiser la variance
        def objective(w):
            return w.T @ Sigma @ w
        
        # Contraintes
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Budget
            {'type': 'eq', 'fun': lambda w: mu.T @ w - target_mu}  # Rendement cible
        ]
        
        # Bornes: pas de short selling
        bounds = [(0, 1) for _ in range(n)]
        
        # Point de départ
        x0 = np.ones(n) / n
        
        # Optimisation
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            w = result.x
            portfolios.append(w)
            
            port_return = mu.T @ w
            port_risk = np.sqrt(w.T @ Sigma @ w)
            
            frontier_returns.append(port_return)
            frontier_risks.append(port_risk)
    
    return np.array(portfolios), np.array(frontier_returns), np.array(frontier_risks)

# =============================
# PORTEFEUILLES REMARQUABLES
# =============================

def minimum_variance_portfolio(returns: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Calcule le portefeuille à variance minimale"""
    Sigma = np.cov(returns)
    mu = np.mean(returns, axis=1)
    
    ones = np.ones(len(mu))
    Sigma_inv = np.linalg.inv(Sigma)
    
    # Formule analytique GMV
    w_gmv = Sigma_inv @ ones / (ones.T @ Sigma_inv @ ones)
    
    # Projection sur les contraintes de positivité
    w_gmv = np.maximum(w_gmv, 0)
    w_gmv = w_gmv / np.sum(w_gmv)
    
    mu_gmv = mu.T @ w_gmv
    sigma_gmv = np.sqrt(w_gmv.T @ Sigma @ w_gmv)
    
    return w_gmv, mu_gmv, sigma_gmv

def maximum_sharpe_portfolio(returns: np.ndarray, risk_free_rate: float = 0.02) -> Tuple[np.ndarray, float, float, float]:
    """Calcule le portefeuille à Sharpe maximal"""
    Sigma = np.cov(returns)
    mu = np.mean(returns, axis=1)
    
    # Rendements en excès
    excess_returns = mu - risk_free_rate / 252  # Conversion journalière
    
    Sigma_inv = np.linalg.inv(Sigma)
    
    # Formule du portefeuille tangent
    numerator = Sigma_inv @ excess_returns
    denominator = np.ones(len(mu)).T @ numerator
    
    w_tangent = numerator / denominator
    
    # Projection sur les contraintes
    w_tangent = np.maximum(w_tangent, 0)
    w_tangent = w_tangent / np.sum(w_tangent)
    
    mu_tangent = mu.T @ w_tangent
    sigma_tangent = np.sqrt(w_tangent.T @ Sigma @ w_tangent)
    sharpe_ratio = (mu_tangent - risk_free_rate / 252) / sigma_tangent
    
    return w_tangent, mu_tangent, sigma_tangent, sharpe_ratio

def maximum_volatility_portfolio(returns: np.ndarray, target_vol: float = 0.12) -> Tuple[np.ndarray, float, float]:
    """
    Construit un portefeuille agressif sur la frontière efficiente
    en visant une volatilité cible (par défaut 12% annuel).
    """
    # Récupérer la frontière efficiente
    if USE_CVXOPT:
        portfolios, frontier_returns, frontier_risks = optimal_portfolio_cvxopt(returns)
    else:
        portfolios, frontier_returns, frontier_risks = optimal_portfolio_scipy(returns)

    # Conversion en volatilité annuelle
    risks_annual = frontier_risks * np.sqrt(252)

    # Trouver l'indice le plus proche de la cible
    idx = np.argmin(np.abs(risks_annual - target_vol))

    w_target = portfolios[idx]
    mu_target = frontier_returns[idx]
    sigma_target = frontier_risks[idx]

    return w_target, mu_target, sigma_target


# =============================
# VISUALISATIONS
# =============================

def create_visualizations(returns: np.ndarray, asset_names: List[str]):
    """
    Crée tous les graphiques Markowitz
    """
    print("\n📊 Génération des visualisations...")
    
    # Calculs principaux
    random_means, random_stds = generate_random_portfolios(returns, n_portfolios=1500)
    
    if USE_CVXOPT:
        portfolios, frontier_returns, frontier_risks = optimal_portfolio_cvxopt(returns)
        print(f"✅ Frontière efficiente calculée avec cvxopt: {len(frontier_returns)} points")
    else:
        portfolios, frontier_returns, frontier_risks = optimal_portfolio_scipy(returns)
        print(f"✅ Frontière efficiente calculée avec scipy: {len(frontier_returns)} points")
    
    # Portefeuilles remarquables
    w_gmv, mu_gmv, sigma_gmv = minimum_variance_portfolio(returns)
    w_tangent, mu_tangent, sigma_tangent, sharpe_max = maximum_sharpe_portfolio(returns)
    w_vol, mu_vol, sigma_vol = maximum_volatility_portfolio(returns)
    
    # Conversion en rendements/risques annuels pour l'affichage
    mu_gmv_annual = mu_gmv * 252
    sigma_gmv_annual = sigma_gmv * np.sqrt(252)
    mu_tangent_annual = mu_tangent * 252
    sigma_tangent_annual = sigma_tangent * np.sqrt(252)
    mu_vol_annual = mu_vol * 252
    sigma_vol_annual = sigma_vol * np.sqrt(252)
    
    frontier_returns_annual = frontier_returns * 252
    frontier_risks_annual = frontier_risks * np.sqrt(252)
    random_means_annual = random_means * 252
    random_stds_annual = random_stds * np.sqrt(252)
    
    # ========== FIGURE PRINCIPALE ==========
    fig = plt.figure(figsize=(18, 12))
    
    # ========== GRAPHIQUE 1: FRONTIÈRE EFFICIENTE ==========
    ax1 = plt.subplot(2, 3, 1)
    
    ax1.scatter(random_stds_annual, random_means_annual, 
               c='lightblue', alpha=0.6, s=10, label='Portefeuilles aléatoires')
    ax1.plot(frontier_risks_annual, frontier_returns_annual, 
            'b-', linewidth=3, label='Frontière Efficiente')
    
    ax1.plot(sigma_gmv_annual, mu_gmv_annual, 'ro', markersize=10, 
            label=f'GMV (σ={sigma_gmv_annual:.1%})')
    ax1.plot(sigma_tangent_annual, mu_tangent_annual, 'go', markersize=10, 
            label=f'Max Sharpe ({sharpe_max*np.sqrt(252):.2f})')
    ax1.plot(sigma_vol_annual, mu_vol_annual, 'mo', markersize=10, 
            label=f'Max Vol (σ={sigma_vol_annual:.1%})')
    
    ax1.set_xlabel('Risque (volatilité annuelle)')
    ax1.set_ylabel('Rendement espéré (annuel)')
    ax1.set_title('Frontière Efficiente de Markowitz')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== GRAPHIQUE 2: ALLOCATION GMV ==========
    ax2 = plt.subplot(2, 3, 2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax2.pie(w_gmv, labels=asset_names, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Allocation - GMV\n(μ={mu_gmv_annual:.1%}, σ={sigma_gmv_annual:.1%})')
    
    # ========== GRAPHIQUE 3: ALLOCATION TANGENT ==========
    ax3 = plt.subplot(2, 3, 3)
    ax3.pie(w_tangent, labels=asset_names, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title(f'Allocation - Max Sharpe\n(μ={mu_tangent_annual:.1%}, σ={sigma_tangent_annual:.1%})')
    
    # ========== GRAPHIQUE 4: ALLOCATION MAX VOL ==========
    ax4 = plt.subplot(2, 3, 4)
    ax4.pie(w_vol, labels=asset_names, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title(f'Allocation - Haute Volatilité\n(μ={mu_vol_annual:.1%}, σ={sigma_vol_annual:.1%})')
    
    # ========== GRAPHIQUE 5: RATIOS RENDEMENT/RISQUE ==========
    ax5 = plt.subplot(2, 3, 5)
    if len(frontier_returns) > 0:
        risk_free_annual = 0.02
        sharpe_ratios = [(r - risk_free_annual) / vol for r, vol in zip(frontier_returns_annual, frontier_risks_annual)]
        
        ax5.plot(frontier_risks_annual, sharpe_ratios, 'b-', linewidth=2)
        ax5.axhline(y=sharpe_max*np.sqrt(252), color='red', linestyle='--', 
                   label=f'Sharpe Max = {sharpe_max*np.sqrt(252):.2f}')
        ax5.plot(sigma_tangent_annual, sharpe_max*np.sqrt(252), 'go', markersize=8)
        
        ax5.set_xlabel('Risque (volatilité annuelle)')
        ax5.set_ylabel('Ratio de Sharpe')
        ax5.set_title('Ratios de Sharpe le long de la Frontière')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # ========== GRAPHIQUE 6: STATISTIQUES ==========
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    data_summary = {
        'Portefeuille': ['GMV', 'Max Sharpe', 'Max Vol'],
        'Rendement': [f'{mu_gmv_annual:.1%}', 
                      f'{mu_tangent_annual:.1%}', 
                      f'{mu_vol_annual:.1%}'],
        'Risque': [f'{sigma_gmv_annual:.1%}', 
                   f'{sigma_tangent_annual:.1%}', 
                   f'{sigma_vol_annual:.1%}'],
        'Sharpe': ['N/A', 
                   f'{sharpe_max*np.sqrt(252):.2f}', 
                   'N/A']
    }
    
    df_summary = pd.DataFrame(data_summary)
    table = ax6.table(cellText=df_summary.values,
                     colLabels=df_summary.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    ax6.set_title('Résumé des Portefeuilles Optimaux', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Résumé console
    print("\n📋 RÉSUMÉ DES RÉSULTATS")
    print("=" * 50)
    print(f"📊 Portefeuilles aléatoires générés: {len(random_means)}")
    print(f"🎯 Points sur la frontière efficiente: {len(frontier_returns)}")
    
    print(f"\n📉 GMV:")
    print(f"   - Rendement: {mu_gmv_annual:.2%}")
    print(f"   - Risque: {sigma_gmv_annual:.2%}")
    print(f"   - Allocation: {dict(zip(asset_names, (w_gmv*100).round(1)))}")
    
    print(f"\n📈 Max Sharpe:")
    print(f"   - Rendement: {mu_tangent_annual:.2%}")
    print(f"   - Risque: {sigma_tangent_annual:.2%}")
    print(f"   - Sharpe: {sharpe_max*np.sqrt(252):.3f}")
    print(f"   - Allocation: {dict(zip(asset_names, (w_tangent*100).round(1)))}")
    
    print(f"\n⚡ Haute Volatilité:")
    print(f"   - Rendement: {mu_vol_annual:.2%}")
    print(f"   - Risque: {sigma_vol_annual:.2%}")
    print(f"   - Allocation: {dict(zip(asset_names, (w_vol*100).round(1)))}")

# =============================
# FONCTION PRINCIPALE
# =============================

def main():
    """Fonction principale d'exécution"""
    print("🚀 DÉMARRAGE: Modèle de Markowitz Simple")
    print("=" * 50)
    
    mu_annual, Sigma_annual, asset_names = get_market_data()
    
    print(f"📊 Actifs: {asset_names}")
    print(f"📈 Rendements annuels: {(mu_annual*100).round(1)}%")
    print(f"📉 Volatilités annuelles: {(np.sqrt(np.diag(Sigma_annual))*100).round(1)}%")
    
    returns_daily = simulate_returns(mu_annual, Sigma_annual, n_obs=252)
    print(f"✅ Simulation: {returns_daily.shape[1]} observations journalières")
    
    create_visualizations(returns_daily, asset_names)
    return returns_daily, asset_names

# =============================
# EXÉCUTION INDÉPENDANTE
# =============================

if __name__ == "__main__":
    print("Modele de Markowitz - Version Simplifiee")
    print("Utilise cvxopt si disponible, sinon scipy")
    print("=" * 60)
    
    returns, assets = main()
    
    print("\nTERMINE: Analyse Markowitz complete")
    print("6 graphiques generes")
    print("Statistiques affichees dans la console")
    print("\n" + "="*60)
