"""
Solution analytique Davis-Norman (chap. 3.4 + 5.2)
- Frontières b(w), s(w) approx.
- Fonction valeur CRRA avec smooth pasting
- Graphiques : frontières, régions, valeur
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------- Paramètres -----------------
mu = 0.07        # drift action
sigma = 0.2      # volatilité action
r = 0.02         # taux sans risque
gamma = 5.0      # aversion CRRA
lam = 0.003      # coût proportionnel

W0 = 1.0         # richesse initiale

# ----------------- Optimum frictionless (Merton) -----------------
pi_star = (mu - r) / (gamma * sigma**2)
print(f"π* (Merton) = {pi_star:.4f}")

# ----------------- Approximation des frontières -----------------
# Résultat Davis-Norman : écart ∝ λ^(1/3)
c = ( (3/2) * pi_star**2 * (1 - pi_star)**2 * gamma * sigma**2 )**(1/3)
delta = c * lam**(1/3)

b = max(0.0, pi_star - delta)
s = min(1.0, pi_star + delta)

print(f"Frontière basse b = {b:.4f}")
print(f"Frontière haute s = {s:.4f}")

# ----------------- Fonction valeur -----------------
def u_crra(w):
    if gamma == 1:
        return np.log(w)
    return w**(1-gamma) / (1-gamma)

def value_function(W, w):
    """
    Fonction valeur par morceaux :
    - Si w ∈ [b,s], valeur frictionless.
    - Sinon, ajustée par coût jusqu'à la frontière.
    """
    # Valeur frictionless
    V0 = u_crra(W) * np.exp(( (mu - r)**2 / (2*gamma*sigma**2) + r ) )
    
    if w < b:
        # si trop bas, il faut acheter → payer coût
        cost = lam * (b - w)
        return (1-cost)**(1-gamma) * V0
    elif w > s:
        # si trop haut, il faut vendre → payer coût
        cost = lam * (w - s)
        return (1-cost)**(1-gamma) * V0
    else:
        return V0

# ----------------- Graphiques -----------------
# 1) Frontières et régions
w_vals = np.linspace(0,1,200)
region = np.where((w_vals>=b)&(w_vals<=s), "Inaction", "Trade")

plt.figure(figsize=(8,4))
plt.axvline(b,color='g',ls='--',label=f"b={b:.3f}")
plt.axvline(s,color='r',ls='--',label=f"s={s:.3f}")
plt.fill_between(w_vals,0,1,where=(w_vals>=b)&(w_vals<=s),alpha=0.2,color="green",label="Zone inaction")
plt.fill_between(w_vals,0,1,where=(w_vals<b)|(w_vals>s),alpha=0.1,color="red",label="Zone trade")
plt.title("Frontières optimales Davis-Norman")
plt.xlabel("Poids en actions w")
plt.ylabel("Zone")
plt.legend()
plt.show()

# 2) Fonction valeur 3D
from mpl_toolkits.mplot3d import Axes3D

W_grid = np.linspace(0.5,2.0,40)
w_grid = np.linspace(0,1,40)
WW, ww = np.meshgrid(W_grid,w_grid)
VV = np.zeros_like(WW)

for i in range(len(W_grid)):
    for j in range(len(w_grid)):
        VV[j,i] = value_function(WW[j,i], ww[j,i])

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(ww,WW,VV,cmap="viridis",alpha=0.8)
ax.set_xlabel("w (poids en actions)")
ax.set_ylabel("W (richesse)")
ax.set_zlabel("V(W,w)")
ax.set_title("Fonction valeur avec bandes Davis-Norman")
plt.show()
