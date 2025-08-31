"""
discrete_bands.py
- Modèle discret multi-périodes (1 actif risqué + cash) avec coûts proportionnels λ
- Investisseur CRRA : u(w)=w^(1-γ)/(1-γ)
- Cible de Merton π* (approx. frictionless) ; on ne trade que si poids sort de [π*±δ]
- On balaie δ pour trouver le meilleur en utilité espérée (grid search rapide)

Usage:
  python discrete_bands.py
"""

import numpy as np

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
    if g == 1.0:
        return np.log(np.maximum(w, 1e-12))
    return (np.maximum(w, 1e-12) ** (1.0 - g)) / (1.0 - g)

def simulate_terminal_utility(delta):
    # portefeuille: (cash, actions). On pilote via poids en actions
    W = np.full(n_paths, W0)
    S = np.ones(n_paths)  # prix normalisé à 1
    w_stock = np.full(n_paths, pi_star)  # on démarre à la cible
    shares = (W * w_stock) / S
    cash = W - shares * S

    for _ in range(N):
        # évolution prix
        Z = rng.standard_normal(n_paths)
        S *= np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z)

        # valeur portefeuille avant rebalancement
        W = cash * np.exp(r * dt) + shares * S
        w_stock = (shares * S) / np.maximum(W, 1e-12)

        # bande de non-transaction
        low, high = pi_star - delta, pi_star + delta
        need_trade = (w_stock < low) | (w_stock > high)
        if not np.any(need_trade):
            cash = cash * np.exp(r * dt)  # seulement rémunération du cash
            continue

        # poids cible = projeter w_stock dans [low, high] vers la borne la + proche
        target_w = np.clip(w_stock, low, high)

        # quantité d'actions désirée
        desired_shares = target_w * W / S
        d_shares = desired_shares - shares

        # coûts proportionnels sur le nominal transigé
        trade_nominal = np.abs(d_shares) * S
        fee = lam * trade_nominal

        # exécution: on paie le coût via le cash
        cash = cash * np.exp(r * dt) - d_shares * S - fee
        shares = desired_shares

        # mise à jour richesse après trades
        W = cash + shares * S

    return u_crra(W, gamma).mean(), W.mean(), W.std()

best = None
for d in delta_grid:
    m_u, m_W, s_W = simulate_terminal_utility(d)
    score = m_u  # on maximise l'utilité moyenne
    if (best is None) or (score > best[0]):
        best = (score, d, m_W, s_W)

print(f"[RESULT] pi*={pi_star:.3f},  best δ={best[1]:.3f}")
print(f"         E[U_T]={best[0]:.6f},   E[W_T]={best[2]:.2f},  Std[W_T]={best[3]:.2f}")
