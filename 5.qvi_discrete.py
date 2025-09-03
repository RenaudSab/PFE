import numpy as np

T = 1.0; N = 64; dt = T / N
mu = 0.07; sigma = 0.2; r = 0.02
lam = 0.003        # coût proportionnel
gamma = 5.0        # aversion CRRA
n_mc = 2000
rng = np.random.default_rng(123)

Wgrid = np.linspace(0.05, 0.95, 51)   # états
Cgrid = np.linspace(0.05, 0.95, 21)   # cibles
B_rel = float(np.exp(r * dt))         # rendement obligataire par pas

print("Paramètres:",
      f"T={T}, N={N}, dt={dt:.4f}, μ={mu}, σ={sigma}, r={r}, λ={lam}, γ={gamma}",
      f"| |W|={len(Wgrid)} [{Wgrid[0]:.2f},{Wgrid[-1]:.2f}]",
      f"| |C|={len(Cgrid)} [{Cgrid[0]:.2f},{Cgrid[-1]:.2f}]", sep="\n")

def u_crra(w):
    w = np.maximum(w, 1e-12)
    if gamma == 1.0:
        return np.log(w)
    return w**(1.0 - gamma) / (1.0 - gamma)

def interp_V(w_points, V_next):
    w = np.clip(w_points, Wgrid[0], Wgrid[-1])
    i = np.searchsorted(Wgrid, w) - 1
    i = np.clip(i, 0, len(Wgrid) - 2)
    x0 = Wgrid[i]; x1 = Wgrid[i + 1]
    a = (w - x0) / (x1 - x0)
    return (1 - a) * V_next[i] + a * V_next[i + 1]

S_rel = np.exp((mu - 0.5*sigma*sigma)*dt
               + sigma*np.sqrt(dt)*rng.standard_normal((N, n_mc)))

V = np.zeros((N + 1, len(Wgrid)))
V[N, :] = u_crra(1.0)

print("Extrait V_N(w):", [f"{V[N,i]:.6e}" for i in [0, len(Wgrid)//2, -1]])

def step_value(w_from, c_to, V_next, S_rel_t):
    """
    Valeur espérée si on vise la cible c_to depuis w_from au début du pas.
    CRRA => facteur ((1-τ) * M)^(1-γ) appliqué à V_next(w_next).
    """
    c = float(np.clip(c_to, Wgrid[0], Wgrid[-1]))
    tau = lam * abs(c - w_from)
    if tau >= 1.0:
        return -np.inf

    M = c * S_rel_t + (1.0 - c) * B_rel           # multiplicateur de richesse
    w_next = (c * S_rel_t) / np.maximum(M, 1e-16) # poids futur
    Vn = interp_V(w_next, V_next)

    factor = (1.0 - tau)**(1.0 - gamma) * np.maximum(M, 1e-16)**(1.0 - gamma)
    return float(np.mean(factor * Vn))


print("\nDémarrage QVI...")
for t in range(N - 1, -1, -1):
    V_next = V[t + 1, :].copy()
    shocks = S_rel[t]
    V_t = np.empty_like(V_next)

    for i, w in enumerate(Wgrid):

        best = step_value(w, w, V_next, shocks)

        for c in Cgrid:
            if c == w:
                continue
            val = step_value(w, c, V_next, shocks)
            if val > best:
                best = val
        V_t[i] = best

    err = np.max(np.abs(V_t - V[t, :]))
    V[t, :] = V_t
    if (N - t) % 5 == 0 or err < 1e-7:
        print(f"t={t:02d} | ||ΔV||_∞={err:.3e}")
    if err < 1e-7:
        print("→ Convergence atteinte.")
        break

print("\nBande d’inaction à t=0...")
V1 = V[1, :]; shocks0 = S_rel[0]
inaction_mask = []
details = []

for w in Wgrid:
    v_nt = step_value(w, w, V1, shocks0)
    v_best = -np.inf; c_best = None
    for c in Cgrid:
        val = step_value(w, c, V1, shocks0)
        if val > v_best:
            v_best, c_best = val, c
    inaction_mask.append(v_nt >= v_best - 1e-12)
    details.append((w, v_nt, v_best, c_best, v_nt >= v_best - 1e-12))

idx = np.where(np.array(inaction_mask))[0]
if idx.size == 0:
    print("Aucune zone d’inaction (trade partout).")
elif idx.size == len(Wgrid):
    print("Toute la grille est en inaction (coûts trop élevés / faible σ).")
else:

    segs, cur = [], [idx[0]]
    for k in idx[1:]:
        if k == cur[-1] + 1: cur.append(k)
        else: segs.append(cur); cur = [k]
    segs.append(cur)
    seg = max(segs, key=len)
    wL, wH = Wgrid[seg[0]], Wgrid[seg[-1]]
    print(f"Inaction principale: [{wL:.3f}, {wH:.3f}]  (largeur {wH-wL:.3f})",
          f"Points en inaction: {len(seg)}/{len(Wgrid)}", sep="\n")

print("\nDiagnostics (quelques w) :")
for j in [0, len(details)//4, len(details)//2, 3*len(details)//4, -1]:
    w, vnt, vtr, cb, ina = details[j]
    tag = "INACTION" if ina else f"TRADE → c*={cb:.3f}"
    print(f"w={w:.3f} | V_nt={vnt:.6e} | V_tr*={vtr:.6e} | {tag}")
