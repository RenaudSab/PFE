"""
continuous_band_merton.py
- Approximation temps continu: cible de Merton π* et rebalancement par bande fixe δ.
- On utilise un pas fin (Δt petit) pour approcher le continu.
- Donne la CE (certitude équivalente) et la bande optimale δ par recherche grossière.
"""

import numpy as np

T = 1.0
N = 10000        
dt = T / N
mu = 0.07; sigma = 0.2; r = 0.02
lam = 0.003
gamma = 5.0
W0 = 100.0
n_paths = 2000

rng = np.random.default_rng(7)
pi_star = (mu - r) / (gamma * sigma * sigma)
pi_star = np.clip(pi_star, 0.0, 1.0)

def u_crra(w, g):
    if g == 1.0:
        return np.log(np.maximum(w, 1e-12))
    return (np.maximum(w, 1e-12) ** (1.0 - g)) / (1.0 - g)

def run(delta):
    W = np.full(n_paths, W0)
    S = np.ones(n_paths)
    w_stock = np.full(n_paths, pi_star)
    shares = (W * w_stock) / S
    cash = W - shares * S
    for _ in range(N):
        Z = rng.standard_normal(n_paths)
        S *= np.exp((mu - 0.5*sigma*sigma)*dt + sigma*np.sqrt(dt)*Z)
        W = cash*np.exp(r*dt) + shares*S
        w_stock = (shares*S)/np.maximum(W,1e-12)
        low, high = pi_star - delta, pi_star + delta
        need = (w_stock < low) | (w_stock > high)
        if np.any(need):
            target = np.clip(w_stock, low, high)
            desired = target*W/S
            d_sh = desired - shares
            fee = lam * np.abs(d_sh) * S
            cash = cash*np.exp(r*dt) - d_sh*S - fee
            shares = desired
            W = cash + shares*S
        else:
            cash = cash*np.exp(r*dt)
    util = u_crra(W, gamma).mean()
    # certitude équivalente (CE) telle que u(CE)=E[u(W_T)]
    if gamma == 1.0:
        CE = np.exp(util)
    else:
        CE = ( (1.0 - gamma)*util ) ** (1.0/(1.0 - gamma))
    return util, CE

best = None
for d in np.linspace(0.0, 0.25, 26):
    u, ce = run(d)
    if (best is None) or (u > best[0]):
        best = (u, d, ce)
print(f"[CONT] pi*={pi_star:.3f},  δ*={best[1]:.3f},  CE≈{best[2]:.2f}")
