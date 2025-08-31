"""
discrete_bands.py - VERSION CORRIGÉE
- Modèle discret multi-périodes (1 actif risqué + cash) avec coûts proportionnels λ
- Investisseur CRRA : u(w)=w^(1-γ)/(1-γ)
- Cible de Merton π* (approx. frictionless) ; on ne trade que si poids sort de [π*±δ]
- On balaie δ pour trouver le meilleur en utilité espérée (grid search rapide)

CORRECTIONS APPORTÉES:
1. Mise à jour correcte de W quand pas de trade
2. Rebalancement vers π* (pas vers la borne)  
3. Gestion des bandes avec bornes [0,1]
4. Vérification de solvabilité
5. Initialisation cohérente

Usage:
  python discrete_bands.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------- params -----------------------
T = 1.0           # horizon (années)
N = 252           # pas de temps
dt = T / N
mu = 0.07         # drift de l'actif
sigma = 0.2       # volatilité
r = 0.02          # taux sans risque
lam = 0.003       # coût proportionnel (aller-retour ~ 2*lam)
gamma = 5.0       # aversion CRRA
W0 = 100.0        # richesse initiale
n_paths = 20000   # MC

rng = np.random.default_rng(42)

# Cible frictionless (Merton) pour CRRA : pi* = (mu - r)/(gamma*sigma^2)
pi_star = (mu - r) / (gamma * sigma * sigma)
pi_star = np.clip(pi_star, 0.0, 1.0)  # bornes simples (pas d'emprunt/short)

# grille de bandes à tester (demi-largeur δ)
delta_grid = np.linspace(0.0, 0.25, 21)  # de 0% à 25% autour de pi*
# -----------------------------------------------------

def u_crra(w, g):
    """Utilité CRRA avec protection contre w<=0"""
    if g == 1.0:
        return np.log(np.maximum(w, 1e-12))
    return (np.maximum(w, 1e-12) ** (1.0 - g)) / (1.0 - g)

def simulate_terminal_utility(delta):
    """Simulation Monte Carlo avec stratégie de bandes corrigée"""
    
    # portefeuille: (cash, actions). On pilote via poids en actions
    W = np.full(n_paths, W0, dtype=np.float64)
    S = np.ones(n_paths, dtype=np.float64)  # prix normalisé à 1
    w_stock = np.full(n_paths, pi_star, dtype=np.float64)  # on démarre à la cible
    shares = (W * w_stock) / S
    cash = W - shares * S
    
    # Définir les bandes avec bornes [0,1]
    low = max(0.0, pi_star - delta)
    high = min(1.0, pi_star + delta)

    for step in range(N):
        # évolution prix
        Z = rng.standard_normal(n_paths)
        S *= np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z)

        # faire fructifier le cash
        cash *= np.exp(r * dt)
        
        # valeur portefeuille avant rebalancement
        W = cash + shares * S
        
        # éviter division par zéro
        W = np.maximum(W, 1e-12)
        w_stock = (shares * S) / W

        # bande de non-transaction
        need_trade = (w_stock < low) | (w_stock > high)
        
        if not np.any(need_trade):
            continue  # pas de trade nécessaire

        # CORRECTION 1: On rebalance vers pi*, pas vers la borne !
        target_w = np.where(need_trade, pi_star, w_stock)

        # quantité d'actions désirée
        desired_shares = target_w * W / S
        d_shares = desired_shares - shares

        # coûts proportionnels sur le nominal transigé
        trade_nominal = np.abs(d_shares) * S
        fee = lam * trade_nominal

        # CORRECTION 2: Vérifier la solvabilité avant d'exécuter
        required_cash = d_shares * S + fee
        can_afford = cash >= required_cash
        
        # N'exécuter que les trades qu'on peut se permettre
        execute_trade = need_trade & can_afford
        
        if np.any(execute_trade):
            # Appliquer les trades seulement là où c'est possible
            cash_change = np.where(execute_trade, -required_cash, 0.0)
            shares_change = np.where(execute_trade, d_shares, 0.0)
            
            cash += cash_change
            shares += shares_change
            
        # CORRECTION 3: Toujours recalculer W à la fin
        W = cash + shares * S

    return u_crra(W, gamma).mean(), W.mean(), W.std()

# Test et affichage
print(f"Paramètres: μ={mu}, σ={sigma}, r={r}, λ={lam}, γ={gamma}")
print(f"π* (Merton) = {pi_star:.4f}")
print(f"Bandes testées: δ ∈ [0, 0.25]")
print("-" * 50)

# Collect all results for plotting
all_results = []
best = None

for i, d in enumerate(delta_grid):
    m_u, m_W, s_W = simulate_terminal_utility(d)
    score = m_u  # on maximise l'utilité moyenne
    all_results.append((d, score, m_W, s_W))
    
    if (best is None) or (score > best[0]):
        best = (score, d, m_W, s_W)
    
    # Affichage progressif
    if i % 5 == 0:
        print(f"δ={d:.3f}: E[U]={m_u:.6f}, E[W]={m_W:.2f}, σ[W]={s_W:.2f}")

print("-" * 50)
print(f"[RÉSULTAT OPTIMAL]")
print(f"π* = {pi_star:.4f}")
print(f"δ* = {best[1]:.4f}")
print(f"E[U_T] = {best[0]:.6f}")
print(f"E[W_T] = {best[2]:.2f}")
print(f"σ[W_T] = {best[3]:.2f}")

# Plot the results
deltas = [r[0] for r in all_results]
utilities = [r[1] for r in all_results]
wealths = [r[2] for r in all_results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Expected Utility vs Delta
ax1.plot(deltas, utilities, 'b-', linewidth=2, marker='o', markersize=4)
ax1.set_xlabel('Delta (demi-largeur de bande)')
ax1.set_ylabel('Utilité Espérée')
ax1.set_title('Utilité Espérée vs Delta')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=best[1], color='r', linestyle='--', linewidth=1,
           label=f'δ* = {best[1]:.3f}')
ax1.legend()

# Plot 2: Expected Wealth vs Delta
ax2.plot(deltas, wealths, 'g-', linewidth=2, marker='s', markersize=4)
ax2.set_xlabel('Delta (demi-largeur de bande)')
ax2.set_ylabel('Richesse Espérée')
ax2.set_title('Richesse Espérée vs Delta')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=best[1], color='r', linestyle='--', linewidth=1,
           label=f'δ* = {best[1]:.3f}')
ax2.legend()

plt.tight_layout()
plt.show()

# Analyse complémentaire
print("\n" + "="*60)
print("ANALYSE DES RÉSULTATS:")
print("="*60)

# Comparer avec le cas sans coûts (delta très grand)
no_trade_idx = -1  # dernier élément = plus grand delta
print(f"Sans rebalancement (δ={deltas[no_trade_idx]:.3f}):")
print(f"  E[U] = {utilities[no_trade_idx]:.6f}")
print(f"  E[W] = {wealths[no_trade_idx]:.2f}")

# Comparer avec rebalancement continu (delta=0)
continuous_idx = 0
print(f"Rebalancement continu (δ={deltas[continuous_idx]:.3f}):")
print(f"  E[U] = {utilities[continuous_idx]:.6f}")
print(f"  E[W] = {wealths[continuous_idx]:.2f}")

print(f"Optimal (δ={best[1]:.3f}):")
print(f"  E[U] = {best[0]:.6f}")
print(f"  E[W] = {best[2]:.2f}")

# Gain d'utilité
gain_vs_no_trade = (best[0] - utilities[no_trade_idx]) / abs(utilities[no_trade_idx]) * 100
gain_vs_continuous = (best[0] - utilities[continuous_idx]) / abs(utilities[continuous_idx]) * 100

print(f"\nGains relatifs en utilité:")
print(f"  vs sans rebalancement: {gain_vs_no_trade:.2f}%")
print(f"  vs rebalancement continu: {gain_vs_continuous:.2f}%")