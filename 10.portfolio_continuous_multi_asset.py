"""
Portefeuille réaliste multi-actifs — simulation continue avec coûts proportionnels
- Actifs : Actions, Obligations, Matières premières
- Corrélations réalistes
- Contraintes : pas de short, cap de diversification
- Politique : cible dyn. risk-parity (EWMA) + bande de non-transaction
- Graphiques : allocation temporelle, performance, nuage risk-return
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# 0) PARAMÈTRES GÉNÉRAUX
# =========================
SEED = 123
np.random.seed(SEED)

years = 5
steps_per_year = 252
dt = 1.0 / steps_per_year
n_steps = years * steps_per_year

assets = ["Actions", "Obligations", "Commodities"]
d = len(assets)

# Hypothèses de marché (annuelles)
mu_ann = np.array([0.07, 0.025, 0.035])           # dérives
sigma_ann = np.array([0.20, 0.07, 0.15])          # volatilités
rho = np.array([[ 1.0, -0.2,  0.30],
                [-0.2,  1.0,  0.00],
                [ 0.30, 0.00, 1.0]])              # corrélations "raisonnables"

# Coûts proportionnels par actif (aller simple)
lambda_vec = np.array([0.0010, 0.0002, 0.0015])   # 10 bps, 2 bps, 15 bps

# Contraintes
W_MAX = 0.70               # cap diversification (max par actif)
NO_SHORT = True            # pas de poids négatifs
BAND_EPS = 0.02            # bande L1 de non-transaction autour de la cible

# Cible dynamique : EWMA
EWMA_DECAY = 0.94          # RiskMetrics (journalier)
RIDGE = 1e-6               # stabilisation inversion covariance
RISK_AVERSION = 1.0        # pondération de la cible Σ^{-1} μ (si <=0 -> fallback inv-vol)

# Monte Carlo pour nuage risk-return
N_SIMS = 100


def project_simplex_box(y, lb=0.0, ub=0.7, s=1.0, tol=1e-12, max_iter=100):
    """
    Projection de y sur {x : sum x = s, lb <= x_i <= ub}
    Résout x = clip(y - τ, lb, ub) avec bisection sur τ.
    """
    y = np.asarray(y, dtype=float)
    lb_vec = np.full_like(y, lb)
    ub_vec = np.full_like(y, ub)
    # bornes de τ
    lo = np.min(y - ub_vec) - 1.0
    hi = np.max(y - lb_vec) + 1.0
    for _ in range(max_iter):
        tau = 0.5 * (lo + hi)
        x = np.clip(y - tau, lb_vec, ub_vec)
        gap = np.sum(x) - s
        if abs(gap) < tol:
            return x
        if gap > 0:
            lo = tau
        else:
            hi = tau
    return x 

def gbm_paths_rel(mu, sigma, rho, n_steps, dt, n_sims):
    """
    Tire des multiplicateurs relatifs R_{t+1}/R_t pour chaque actif (shape (n_sims, n_steps, d)).
    """
    d = len(mu)
    L = np.linalg.cholesky(rho)
    Z = np.random.randn(n_sims, n_steps, d)
    Z_corr = Z @ L.T
    drift = (mu - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt)
    R_rel = np.exp(drift + diff * Z_corr)  # multiplicateurs
    return R_rel  # (n_sims, n_steps, d)

def ewma_update_cov(Sigma, r, decay):
    """
    Mise à jour EWMA de la covariance (r : vecteur de rendements simples).
    """
    return decay * Sigma + (1.0 - decay) * np.outer(r, r)

def target_weights_from_cov(Sigma, mu, ridge=1e-6):
    """
    Cible "mean-variance" : w ∝ Σ^{-1} μ  (si composantes <=0, fallback inverse-vol).
    Projection ensuite sur le simplexe borné.
    """
    d = len(mu)
    Sig = Sigma + ridge * np.eye(d)
    try:
        y = np.linalg.solve(Sig, mu)
    except np.linalg.LinAlgError:
        y = np.diag(1.0 / np.maximum(np.diag(Sig), 1e-8)) @ mu
    # si tout <=0 -> fallback inverse-vol
    if np.all(y <= 0):
        inv_vol = 1.0 / np.sqrt(np.maximum(np.diag(Sig), 1e-12))
        y = inv_vol
    # pas de short, cap
    y = np.maximum(y, 0.0) if NO_SHORT else y
    w = project_simplex_box(y, lb=0.0 if NO_SHORT else -W_MAX, ub=W_MAX, s=1.0)
    return w


def simulate_one_path(R_rel, mu_ann, ewma_decay=EWMA_DECAY):
    """
    Simule une trajectoire (R_rel : (n_steps, d)) avec :
      - cibles dynamiques (Σ EWMA)
      - bande de non-transaction
      - coûts proportionnels
    Retourne : dict avec wealth, weights, fees, rebalances, port_returns
    """
    n_steps, d = R_rel.shape
    W = 100.0
    wealth = np.empty(n_steps+1); wealth[0] = W

    Sigma = np.diag((sigma_ann/np.sqrt(steps_per_year))**2)

    w = target_weights_from_cov(Sigma, mu_ann)
    weights = np.empty((n_steps+1, d)); weights[0] = w

    total_fees = 0.0
    n_rebals = 0
    port_rets = np.empty(n_steps)

    for t in range(n_steps):
        r_rel = R_rel[t]                 # multiplicateurs actifs (t -> t+1)
        # Richesse brute
        gross_mult = float(np.dot(w, r_rel))
        W *= gross_mult
        wealth[t+1] = W

        # Rendements simples pour EWMA
        r_simple = r_rel - 1.0
        Sigma = ewma_update_cov(Sigma, r_simple, ewma_decay)

        # Poids dérivés par la dérive des prix 
        w_drift = (w * r_rel) / gross_mult

        # Nouvelle cible
        w_tgt = target_weights_from_cov(Sigma, mu_ann)

        # Bande L1
        if np.sum(np.abs(w_drift - w_tgt)) > BAND_EPS:
            # Coût proportionnel sur le nominal échangé
            trade_abs = np.abs(w_tgt - w_drift)
            fee_frac = float(np.dot(lambda_vec, trade_abs))
            W *= (1.0 - fee_frac)              # on paie les frais
            total_fees += fee_frac * wealth[t+1]  
            n_rebals += 1
            w = w_tgt
        else:
            w = w_drift

        weights[t+1] = w
        port_rets[t] = gross_mult - 1.0

    return {
        "wealth": wealth,
        "weights": weights,
        "fees_paid": total_fees,
        "n_rebals": n_rebals,
        "port_returns": port_rets
    }

# Convertir annuels
mu_step = mu_ann
sigma_step = sigma_ann

# 3.1 Une trajectoire de référence (pour tracer l’allocation/wealth)
R_rel_1 = gbm_paths_rel(mu_step, sigma_step, rho, n_steps, dt, n_sims=1)[0]  # (n_steps, d)
res_1 = simulate_one_path(R_rel_1, mu_ann)

# 3.2 Monte Carlo pour nuage risk-return
R_rel_MC = gbm_paths_rel(mu_step, sigma_step, rho, n_steps, dt, n_sims=N_SIMS)
mc_ann_ret = np.empty(N_SIMS)
mc_ann_vol = np.empty(N_SIMS)

for s in range(N_SIMS):
    res = simulate_one_path(R_rel_MC[s], mu_ann)
    r = res["port_returns"]              # quotidiens (après frais)
    mu_a = (1 + np.mean(r))**steps_per_year - 1.0
    vol_a = np.std(r) * np.sqrt(steps_per_year)
    mc_ann_ret[s] = mu_a
    mc_ann_vol[s] = vol_a

# 4.1 Allocation temporelle (stacked area)
plt.figure(figsize=(10,5))
t = np.arange(n_steps+1) / steps_per_year
plt.stackplot(t, res_1["weights"].T, labels=assets, alpha=0.9)
plt.title("Allocation temporelle (poids) — bande de non-transaction")
plt.xlabel("Temps (années)")
plt.ylabel("Poids")
plt.ylim(0,1)
plt.legend(loc="upper left", ncol=3)
plt.grid(alpha=0.25)
plt.tight_layout()

# 4.2 Performance (wealth)
plt.figure(figsize=(10,5))
plt.plot(t, res_1["wealth"], lw=2)
plt.title(f"Évolution de la richesse (frais inclus) — rebalances={res_1['n_rebals']}, frais totaux≈{res_1['fees_paid']:.2f}")
plt.xlabel("Temps (années)")
plt.ylabel("Richesse")
plt.grid(alpha=0.25)
plt.tight_layout()

# 4.3 Nuage risk-return (Monte Carlo)
plt.figure(figsize=(7,6))
plt.scatter(mc_ann_vol, mc_ann_ret, alpha=0.7)
plt.title("Nuage risk-return (après coûts)")
plt.xlabel("Volatilité annualisée")
plt.ylabel("Rendement annualisé")
# Annotation moyenne
plt.scatter(np.mean(mc_ann_vol), np.mean(mc_ann_ret), c="red", s=80, label="Moyenne")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.show()
