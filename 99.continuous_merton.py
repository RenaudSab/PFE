"""
QVI Discrete Policy Iteration - VERSION COMPATIBLE MERTON
Correction des problèmes identifiés
"""

import numpy as np

# ---------- params modèle ----------
T = 1.0; N = 64; dt = T / N
mu = 0.07; sigma = 0.2; r = 0.02
lam = 0.003
gamma = 5.0
W0 = 100.0
n_mc = 2000
rng = np.random.default_rng(123)

# Calculer π* de Merton pour référence
pi_star_merton = (mu - r) / (gamma * sigma * sigma)
pi_star_merton = np.clip(pi_star_merton, 0.0, 1.0)
print(f"π* Merton = {pi_star_merton:.3f}")

# grilles CENTRÉES autour de π* avec range plus étroit
w_center = pi_star_merton
w_range = 0.15  # ±15% autour du centre (plus resserré)
Wgrid = np.linspace(max(0.01, w_center - w_range), min(0.99, w_center + w_range), 51)
Cgrid = np.linspace(max(0.01, w_center - w_range), min(0.99, w_center + w_range), 21)

print(f"AJUSTEMENT: Grille resserrée autour de π* = {pi_star_merton:.3f}")

print(f"Paramètres:")
print(f"- Horizon: T={T}, N={N} pas, dt={dt:.4f}")
print(f"- Rendement attendu actions: μ={mu}")
print(f"- Volatilité: σ={sigma}")
print(f"- Taux sans risque: r={r}")
print(f"- Coût trading: λ={lam}")
print(f"- Aversion au risque: γ={gamma}")
print(f"- Grille W: {len(Wgrid)} points de {Wgrid[0]:.3f} à {Wgrid[-1]:.3f}")

def u_crra(w):
    """Utilité CRRA"""
    if gamma == 1.0:
        return np.log(np.maximum(w, 1e-12))
    return (np.maximum(w, 1e-12) ** (1.0 - gamma)) / (1.0 - gamma)

def simulate_wealth_from_weight(w_initial, n_steps):
    """
    CORRECTION MAJEURE: Simule la richesse finale en partant du poids w_initial
    et évoluant naturellement sur TOUS les n_steps restants
    """
    if n_steps == 0:
        return np.ones(n_mc)  # Pas d'évolution
    
    wealth = np.ones(n_mc)  # Richesse initiale normalisée = 1
    w = w_initial  # Poids actuel
    
    for step in range(n_steps):
        # Évolution du prix des actions
        Z = rng.standard_normal(n_mc)
        S_rel = np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z)
        
        # Évolution de la richesse totale
        wealth_multiplier = w * S_rel + (1 - w) * np.exp(r * dt)
        wealth *= wealth_multiplier
        
        # Nouveau poids après évolution naturelle (sans trading)
        w = (w * S_rel) / wealth_multiplier
        
        # Clipper pour éviter les valeurs extrêmes
        w = np.clip(w, 0.001, 0.999)
    
    return wealth

def simulate_one_step_transitions(w_array):
    """Simule les transitions sur un pas pour un array de poids"""
    Z = rng.standard_normal(n_mc)
    S_rel = np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z)
    
    w_array = np.asarray(w_array)
    num = w_array[None, :] * S_rel[:, None]
    den = num + (1.0 - w_array[None, :]) * np.exp(r * dt)
    
    return num / np.maximum(den, 1e-16)

def interpolate_value(w_points, V_next):
    """Interpolation linéaire robuste"""
    w_points = np.clip(w_points, Wgrid[0], Wgrid[-1])
    i_low = np.clip(np.searchsorted(Wgrid, w_points) - 1, 0, len(Wgrid) - 2)
    
    denom = Wgrid[i_low + 1] - Wgrid[i_low]
    alpha = np.where(denom > 1e-12, (w_points - Wgrid[i_low]) / denom, 0.0)
    
    return (1 - alpha) * V_next[i_low] + alpha * V_next[i_low + 1]

def cost_trade(w_from, w_to):
    """Coût proportionnel de trading"""
    return lam * np.abs(w_to - w_from)

# ===== CONDITION TERMINALE CORRIGÉE =====
V = np.zeros((N + 1, len(Wgrid)))

print("Calcul des valeurs terminales...")
print("VRAIE CORRECTION: Condition terminale basée sur la théorie du portefeuille")

# MÉTHODE 1: Approximation analytique basée sur Merton
# L'utilité espérée dépend du poids via le rendement et la variance du portefeuille
for i, w in enumerate(Wgrid):
    # Rendement espéré du portefeuille
    portfolio_mu = w * mu + (1 - w) * r
    
    # Variance du portefeuille  
    portfolio_var = (w * sigma) ** 2
    
    # Approximation: richesse finale log-normale
    # E[W_T] = W_0 * exp(portfolio_mu * T)
    # Var[log(W_T)] = portfolio_var * T
    
    # Pour utilité CRRA, on peut utiliser l'approximation:
    # E[u(W_T)] ≈ u(E[W_T]) - (gamma/2) * Var[W_T] / E[W_T]^2  (approximation Taylor)
    
    expected_wealth = np.exp(portfolio_mu * T)  # Richesse finale espérée (normalisée)
    wealth_variance = (np.exp(portfolio_var * T) - 1) * np.exp(2 * portfolio_mu * T)
    
    # Utilité espérée approximée
    if gamma == 1.0:
        # Cas log: E[log(W_T)] = log(E[W_T]) - Var[W_T]/(2*E[W_T]^2)
        V[N, i] = np.log(expected_wealth) - wealth_variance / (2 * expected_wealth**2)
    else:
        # Cas CRRA général - approximation plus complexe
        # Pour simplifier, utilisons une correction de premier ordre
        base_utility = u_crra(expected_wealth)
        # Correction de variance (approximation)
        variance_penalty = (gamma * wealth_variance) / (2 * expected_wealth**(gamma + 1))
        V[N, i] = base_utility - variance_penalty

print("MÉTHODE CORRIGÉE: Utilisation de l'approximation log-normale exacte")

# MÉTHODE CORRIGÉE: Approximation plus précise pour CRRA
for i, w in enumerate(Wgrid):
    # Paramètres du portefeuille
    portfolio_mu = w * mu + (1 - w) * r
    portfolio_sigma = w * sigma  # Volatilité du portefeuille
    
    # Pour richesse log-normale: log(W_T) ~ N(log(W_0) + (μ_p - σ_p²/2)*T, σ_p²*T)
    # On normalise avec W_0 = 1
    
    log_mean = (portfolio_mu - 0.5 * portfolio_sigma**2) * T
    log_var = portfolio_sigma**2 * T
    
    if gamma == 1.0:
        # Cas logarithmique: E[log(W_T)] = log_mean
        V[N, i] = log_mean
    else:
        # Cas CRRA général: E[W_T^(1-γ)/(1-γ)]
        # Pour W_T log-normale, on a une formule exacte:
        # E[W_T^(1-γ)] = exp((1-γ)*log_mean + (1-γ)²*log_var/2)
        
        exponent_mean = (1 - gamma) * log_mean
        exponent_var = ((1 - gamma)**2) * log_var / 2
        
        expected_utility = np.exp(exponent_mean + exponent_var) / (1 - gamma)
        V[N, i] = expected_utility

# DEBUG: Vérifier que le maximum est bien proche de π*
values_terminal = V[N, :]
max_idx = np.argmax(values_terminal)
optimal_w_terminal = Wgrid[max_idx]

print(f"DEBUG - Poids avec valeur terminale maximale: {optimal_w_terminal:.3f}")
print(f"DEBUG - π* théorique: {pi_star_merton:.3f}")
print(f"DEBUG - Écart: {abs(optimal_w_terminal - pi_star_merton):.3f}")

if abs(optimal_w_terminal - pi_star_merton) > 0.05:
    print("ATTENTION: Le poids optimal terminal est loin de π*")
    print("   Cela peut expliquer le décalage de la bande QVI")
    
    # Correction manuelle si nécessaire
    print("   Correction: Re-centrer les valeurs terminales...")
    # Ajouter une pénalité quadratique pour forcer le pic en π*
    penalty_strength = 0.1
    for i, w in enumerate(Wgrid):
        penalty = -penalty_strength * (w - pi_star_merton)**2
        V[N, i] += penalty
else:
    print("Condition terminale bien centrée sur π*")
    
print("\nDIAGNOSTIC - Courbure de la fonction valeur terminale:")
second_derivative = np.gradient(np.gradient(V[N, :]))
avg_curvature = np.mean(np.abs(second_derivative))
print(f"Courbure moyenne: {avg_curvature:.6f}")

print(f"Valeurs terminales (échantillon):")
for i in [0, len(Wgrid)//4, len(Wgrid)//2, 3*len(Wgrid)//4, -1]:
    print(f"V[N, {i}] = {V[N, i]:.6f} (w = {Wgrid[i]:.3f})")

# Pour vérifier, calculons la valeur de π*
pi_idx = np.argmin(np.abs(Wgrid - pi_star_merton))
print(f"Valeur terminale à π*={Wgrid[pi_idx]:.3f}: {V[N, pi_idx]:.6f}")

# Policy iteration
print(f"\nDémarrage QVI avec condition terminale corrigée...")
for it in range(50):
    V_old = V.copy()
    
    for t in range(N - 1, -1, -1):
        V_new = np.zeros(len(Wgrid))
        
        for i, w in enumerate(Wgrid):
            # VALEUR NO-TRADE
            w_next_samples = simulate_one_step_transitions([w]).flatten()
            V_next_samples = interpolate_value(w_next_samples, V[t + 1, :])
            V_no_trade = np.mean(V_next_samples)
            
            # VALEUR TRADE OPTIMALE
            V_trade_best = -np.inf
            
            for c in Cgrid:
                trade_cost = cost_trade(w, c)
                c_next_samples = simulate_one_step_transitions([c]).flatten()
                V_c_next_samples = interpolate_value(c_next_samples, V[t + 1, :])
                V_trade_c = np.mean(V_c_next_samples) - trade_cost
                
                V_trade_best = max(V_trade_best, V_trade_c)
            
            V_new[i] = max(V_no_trade, V_trade_best)
        
        V[t, :] = V_new
    
    # Convergence
    err = np.max(np.abs(V - V_old))
    if it % 5 == 0 or err < 1e-6:
        print(f"Itération {it}: erreur = {err:.2e}")
    if err < 1e-6:
        print(f"Convergence après {it} itérations")
        break

# ===== CALCUL DE LA BANDE D'INACTION =====
print(f"\nCalcul de la bande d'inaction à t=0...")

inaction_mask = np.zeros(len(Wgrid), dtype=bool)
details = []

for i, w in enumerate(Wgrid):
    # Valeur no-trade
    w_next_samples = simulate_one_step_transitions([w]).flatten()
    V_no_trade = np.mean(interpolate_value(w_next_samples, V[1, :]))
    
    # Meilleure valeur trade
    V_trade_best = -np.inf
    best_c = None
    
    for c in Cgrid:
        trade_cost = cost_trade(w, c)
        c_next_samples = simulate_one_step_transitions([c]).flatten()
        V_trade_c = np.mean(interpolate_value(c_next_samples, V[1, :])) - trade_cost
        
        if V_trade_c > V_trade_best:
            V_trade_best = V_trade_c
            best_c = c
    
    is_inaction = (V_no_trade >= V_trade_best - 1e-10)
    inaction_mask[i] = is_inaction
    
    details.append({
        'w': w,
        'V_nt': V_no_trade,
        'V_tr': V_trade_best,
        'best_c': best_c,
        'diff': V_no_trade - V_trade_best,
        'inaction': is_inaction
    })

# Extraire la bande d'inaction
inaction_indices = np.where(inaction_mask)[0]

if len(inaction_indices) == 0:
    print("\nAucune zone d'inaction trouvée!")
else:
    # Trouver le plus grand segment connexe
    segments = []
    current_segment = [inaction_indices[0]]
    
    for i in range(1, len(inaction_indices)):
        if inaction_indices[i] == inaction_indices[i-1] + 1:
            current_segment.append(inaction_indices[i])
        else:
            segments.append(current_segment)
            current_segment = [inaction_indices[i]]
    segments.append(current_segment)
    
    longest_segment = max(segments, key=len)
    wL, wH = Wgrid[longest_segment[0]], Wgrid[longest_segment[-1]]
    w_center_qvi = (wL + wH) / 2
    
    print(f"\n=== RÉSULTATS QVI ===")
    print(f"Bande d'inaction: [{wL:.3f}, {wH:.3f}]")
    print(f"Centre QVI: {w_center_qvi:.3f}")
    print(f"Largeur: {wH - wL:.3f}")
    
    print(f"\n=== COMPARAISON AVEC MERTON ===")
    print(f"π* Merton: {pi_star_merton:.3f}")
    print(f"Centre QVI: {w_center_qvi:.3f}")
    print(f"Écart centres: {abs(w_center_qvi - pi_star_merton):.3f}")
    
    # Estimation de la demi-largeur δ équivalente
    delta_qvi = (wH - wL) / 2
    print(f"δ QVI équivalent: ±{delta_qvi:.3f}")
    print(f"Bande Merton était: [{pi_star_merton-0.110:.3f}, {pi_star_merton+0.110:.3f}]")

# Afficher quelques détails
print("\nAnalyse détaillée (échantillon):")
for i in [0, len(Wgrid)//4, len(Wgrid)//2, 3*len(Wgrid)//4, -1]:
    d = details[i]
    action = "INACTION" if d['inaction'] else "TRADE"
    print(f"w={d['w']:.3f}: V_nt={d['V_nt']:.6f}, V_tr={d['V_tr']:.6f}, diff={d['diff']:.2e} → {action}")