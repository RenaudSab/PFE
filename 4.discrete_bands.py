"""
discrete_bands.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------- params -----------------------
T = 1.0           # horizon (années)
N = 252           # pas de temps
dt = T / N
mu = 0.07         
sigma = 0.2       # volatilité
r = 0.02          # taux sans risque
lam = 0.003       # coût proportionnel (aller-retour ~ 2*lam)
gamma = 5.0       # aversion CRRA
W0 = 100.0        # richesse initiale
n_paths = 50000   #Monte Carlo

pi_star = (mu - r) / (gamma * sigma * sigma)
pi_star = np.clip(pi_star, 0.0, 1.0)

# grille de bandes (demi-largeur δ)
delta_grid = np.linspace(0.0, 0.25, 26)  
# -----------------------------------------------------

def u_crra(w, g):
    """Utilité CRRA avec protection contre w<=0 du au log"""
    if g == 1.0:
        return np.log(np.maximum(w, 1e-12))
    return (np.maximum(w, 1e-12) ** (1.0 - g)) / (1.0 - g)

def simulate_terminal_utility(delta, seed_base=42):
    """Simulation Monte Carlo a pour toutes les deltas"""
    rng = np.random.default_rng(seed_base)
    
    W = np.full(n_paths, W0, dtype=np.float64)
    S = np.ones(n_paths, dtype=np.float64)
    w_stock = np.full(n_paths, pi_star, dtype=np.float64)
    shares = (W * w_stock) / S
    cash = W - shares * S
    
    # Définir les bandes avec bornes [0,1]
    low = max(0.0, pi_star - delta)
    high = min(1.0, pi_star + delta)
    
    # Compteurs pour diagnostics
    total_trades = 0
    total_fees = 0.0

    for step in range(N):
        # évolution prix
        Z = rng.standard_normal(n_paths)
        S *= np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z)

        # faire fructifier le cash
        cash *= np.exp(r * dt)
        
        # valeur portefeuille avant rebalancement
        W = cash + shares * S
        W = np.maximum(W, 1e-12)
        w_stock = (shares * S) / W

        # bande de non-transaction
        need_trade = (w_stock < low) | (w_stock > high)
        
        if not np.any(need_trade):
            continue

        total_trades += np.sum(need_trade)
        
        # Rebalancement vers pi* (pas vers la borne)
        target_w = np.where(need_trade, pi_star, w_stock)

        # quantité d'actions désirée
        desired_shares = target_w * W / S
        d_shares = desired_shares - shares

        # coûts proportionnels
        trade_nominal = np.abs(d_shares) * S
        fee = lam * trade_nominal
        total_fees += np.sum(fee)

        # Vérifier la solvabilité
        required_cash = d_shares * S + fee
        can_afford = cash >= required_cash
        execute_trade = need_trade & can_afford
        
        if np.any(execute_trade):
            cash_change = np.where(execute_trade, -required_cash, 0.0)
            shares_change = np.where(execute_trade, d_shares, 0.0)
            cash += cash_change
            shares += shares_change
        
        W = cash + shares * S

    avg_trades = total_trades / n_paths
    avg_fees = total_fees / n_paths
    
    return u_crra(W, gamma).mean(), W.mean(), W.std(), avg_trades, avg_fees

# Simulation
print(f"Paramètres: μ={mu}, σ={sigma}, r={r}, λ={lam}, γ={gamma}")
print(f"π* (Merton) = {pi_star:.4f}")
print(f"Simulations: {n_paths:,} chemins")
print(f"Bandes testées: δ ∈ [0, 0.25] ({len(delta_grid)} points)")
print("-" * 70)

all_results = []
best = None

for i, d in enumerate(delta_grid):
    m_u, m_W, s_W, trades, fees = simulate_terminal_utility(d, seed_base=42)
    score = m_u
    all_results.append((d, score, m_W, s_W, trades, fees))
    
    if (best is None) or (score > best[0]):
        best = (score, d, m_W, s_W, trades, fees)
    
    if i % 5 == 0:
        print(f"δ={d:.3f}: E[U]={m_u:.6e}, E[W]={m_W:.2f}, "
              f"σ[W]={s_W:.2f}, Trades={trades:.1f}, Fees={fees:.4f}")

print("-" * 70)
print(f"[RÉSULTAT OPTIMAL]")
print(f"π* = {pi_star:.4f}")
print(f"δ* = {best[1]:.4f}")
print(f"E[U_T] = {best[0]:.8e}") 
print(f"E[W_T] = {best[2]:.2f}")
print(f"σ[W_T] = {best[3]:.2f}")
print(f"Trades/chemin = {best[4]:.1f}")
print(f"Frais/chemin = {best[5]:.4f}")

# Graphiques
deltas = [r[0] for r in all_results]
utilities = [r[1] for r in all_results]
wealths = [r[2] for r in all_results]
trades_list = [r[4] for r in all_results]
fees_list = [r[5] for r in all_results]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Utilité 
ax1.plot(deltas, utilities, 'b-', linewidth=2, marker='o', markersize=4)
ax1.set_xlabel('Delta (demi-largeur de bande)')
ax1.set_ylabel('Utilité Espérée')
ax1.set_title(f'Utilité Espérée vs Delta')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=best[1], color='r', linestyle='--', linewidth=1,
           label=f'δ* = {best[1]:.3f}')
ax1.legend()

# Plot 2: Richesse
ax2.plot(deltas, wealths, 'g-', linewidth=2, marker='s', markersize=4)
ax2.set_xlabel('Delta (demi-largeur de bande)')
ax2.set_ylabel('Richesse Espérée')
ax2.set_title('Richesse Espérée vs Delta')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=best[1], color='r', linestyle='--', linewidth=1,
           label=f'δ* = {best[1]:.3f}')
ax2.legend()

# Plot 3: Fréquence de trading
ax3.plot(deltas, trades_list, 'r-', linewidth=2, marker='^', markersize=4)
ax3.set_xlabel('Delta (demi-largeur de bande)')
ax3.set_ylabel('Trades par chemin')
ax3.set_title('Activité de Trading vs Delta')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')  # Échelle log car ça va de 0 à 252

# Plot 4: Coûts de transaction
ax4.plot(deltas, fees_list, 'm-', linewidth=2, marker='d', markersize=4)
ax4.set_xlabel('Delta (demi-largeur de bande)')
ax4.set_ylabel('Frais par chemin')
ax4.set_title('Coûts de Transaction vs Delta')
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')  # Échelle log

plt.tight_layout()
plt.show()

# Analyse économique
print("\n" + "="*70)
print("ANALYSE ÉCONOMIQUE:")
print("="*70)

# Points de référence
rebalance_continu = all_results[0]  # delta = 0
no_rebalance = all_results[-1]     # delta max

print(f"Rebalancement continu (δ=0):")
print(f"  E[U] = {rebalance_continu[1]:.6e}")
print(f"  E[W] = {rebalance_continu[2]:.2f}")
print(f"  Trades = {rebalance_continu[4]:.0f}/chemin")
print(f"  Frais = {rebalance_continu[5]:.4f}/chemin")

print(f"\nPas de rebalancement (δ={no_rebalance[0]:.2f}):")
print(f"  E[U] = {no_rebalance[1]:.6e}")
print(f"  E[W] = {no_rebalance[2]:.2f}")
print(f"  Trades = {no_rebalance[4]:.1f}/chemin")
print(f"  Frais = {no_rebalance[5]:.4f}/chemin")

print(f"\nOptimal (δ*={best[1]:.3f}):")
print(f"  E[U] = {best[0]:.6e}")
print(f"  E[W] = {best[2]:.2f}")
print(f"  Trades = {best[4]:.1f}/chemin")
print(f"  Frais = {best[5]:.4f}/chemin")

# Gains relatifs
gain_vs_continuous = (best[0] - rebalance_continu[1]) / abs(rebalance_continu[1]) * 100
gain_vs_no_rebal = (best[0] - no_rebalance[1]) / abs(no_rebalance[1]) * 100

print(f"\nGains d'utilité (en %):")
print(f"  vs rebalancement continu: {gain_vs_continuous:+.2f}%")
print(f"  vs pas de rebalancement:  {gain_vs_no_rebal:+.2f}%")

print(f"\nRéduction des coûts:")
print(f"  vs rebalancement continu: {(1-best[5]/rebalance_continu[5])*100:.1f}%")