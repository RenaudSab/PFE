"""
QVI Discrete Policy Iteration - VERSION CORRIGÉE
Le problème était dans la condition terminale !
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

# grilles
Wgrid = np.linspace(0.05, 0.95, 51)  # Éviter les bords 0 et 1 problématiques
Cgrid = np.linspace(0.05, 0.95, 21)

print(f"Paramètres:")
print(f"- Horizon: T={T}, N={N} pas, dt={dt:.4f}")
print(f"- Rendement attendu actions: μ={mu}")
print(f"- Volatilité: σ={sigma}")
print(f"- Taux sans risque: r={r}")
print(f"- Coût trading: λ={lam}")
print(f"- Aversion au risque: γ={gamma}")
print(f"- Grille W: {len(Wgrid)} points de {Wgrid[0]:.2f} à {Wgrid[-1]:.2f}")

def u_crra(w):
    """Utilité CRRA"""
    if gamma == 1.0:
        return np.log(np.maximum(w, 1e-12))
    return (np.maximum(w, 1e-12) ** (1.0 - gamma)) / (1.0 - gamma)

def simulate_wealth_terminal(w_initial, n_steps_remaining):
    """
    Simule la richesse terminale en partant d'un poids w_initial
    et en évoluant naturellement (sans trading) sur n_steps_remaining pas.
    """
    if n_steps_remaining == 0:
        return np.ones(n_mc)  # Richesse normalisée = 1
    
    # Évolution sur tous les pas restants
    wealth = np.ones(n_mc)  # Richesse initiale normalisée
    w = w_initial
    
    for step in range(n_steps_remaining):
        Z = rng.standard_normal(n_mc)
        S_rel = np.exp((mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z)
        
        # Évolution de la richesse: W_new = W * (w*S_rel + (1-w)*exp(r*dt))
        wealth_multiplier = w * S_rel + (1 - w) * np.exp(r * dt)
        wealth *= wealth_multiplier
        
        # Nouveau poids (évolution naturelle)
        w = (w * S_rel) / wealth_multiplier
    
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
    """Interpolation linéaire"""
    w_points = np.clip(w_points, Wgrid[0], Wgrid[-1])  # Clipper dans la grille
    i_low = np.clip(np.searchsorted(Wgrid, w_points) - 1, 0, len(Wgrid) - 2)
    alpha = (w_points - Wgrid[i_low]) / (Wgrid[i_low + 1] - Wgrid[i_low])
    return (1 - alpha) * V_next[i_low] + alpha * V_next[i_low + 1]

def cost_trade(w_from, w_to):
    """Coût proportionnel de trading"""
    return lam * np.abs(w_to - w_from)

# ===== CONDITION TERMINALE CORRECTE =====
V = np.zeros((N + 1, len(Wgrid)))

# Pour chaque poids terminal, calculer l'utilité espérée de la richesse finale
print("Calcul des valeurs terminales...")
for i, w in enumerate(Wgrid):
    # Partir du poids w, ne pas trader, et voir la richesse finale
    # Comme on est déjà en T, pas d'évolution supplémentaire
    V[N, i] = u_crra(1.0)  # Richesse = 1 (normalisée)

# Mais le vrai calcul doit intégrer le fait que différents poids donnent
# différentes distributions de richesse finale ! 
# Recalculons correctement :

for i, w in enumerate(Wgrid):
    # Si on termine avec le poids w, quelle est l'utilité espérée ?
    # On doit considérer que la richesse finale dépend de l'évolution depuis t=0
    # Pour simplifier, on suppose qu'on évalue l'utilité du "rendement" final
    
    # Approximation : utilité basée sur le rendement espéré
    expected_return = w * (np.exp(mu * dt) - 1) + (1 - w) * (np.exp(r * dt) - 1)
    final_wealth = 1.0 + expected_return  # Approximation premier ordre
    V[N, i] = u_crra(final_wealth)

print(f"Valeurs terminales (échantillon):")
for i in [0, len(Wgrid)//4, len(Wgrid)//2, 3*len(Wgrid)//4, -1]:
    print(f"V[N, {i}] = {V[N, i]:.6f} (w = {Wgrid[i]:.3f})")

# Policy iteration
print(f"\nDémarrage QVI...")
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

# Afficher quelques détails
print("\nAnalyse détaillée (échantillon):")
for i in [0, len(Wgrid)//4, len(Wgrid)//2, 3*len(Wgrid)//4, -1]:
    d = details[i]
    action = "INACTION" if d['inaction'] else "TRADE"
    print(f"w={d['w']:.3f}: V_nt={d['V_nt']:.6f}, V_tr={d['V_tr']:.6f}, diff={d['diff']:.2e} → {action}")

# Extraire la bande d'inaction
inaction_indices = np.where(inaction_mask)[0]

if len(inaction_indices) == 0:
    print("\nAucune zone d'inaction trouvée!")
elif len(inaction_indices) == len(Wgrid):
    print(f"\nToute la grille est en inaction - vérifier les paramètres!")
    print(f"Suggestion: augmenter λ (coût trading) ou réduire γ (aversion risque)")
else:
    # Trouver les segments connexes
    segments = []
    current_segment = [inaction_indices[0]]
    
    for i in range(1, len(inaction_indices)):
        if inaction_indices[i] == inaction_indices[i-1] + 1:
            current_segment.append(inaction_indices[i])
        else:
            segments.append(current_segment)
            current_segment = [inaction_indices[i]]
    segments.append(current_segment)
    
    # Plus long segment
    longest_segment = max(segments, key=len)
    wL, wH = Wgrid[longest_segment[0]], Wgrid[longest_segment[-1]]
    
    print(f"\n=== RÉSULTATS ===")
    print(f"Bande d'inaction principale: [{wL:.3f}, {wH:.3f}]")
    print(f"Largeur: {wH - wL:.3f}")
    print(f"Points en inaction: {len(longest_segment)}/{len(Wgrid)}")
    
    if len(segments) > 1:
        print(f"Nombre de segments d'inaction: {len(segments)}")