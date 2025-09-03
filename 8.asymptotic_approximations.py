"""
Approximations asymptotiques (section 3.6, Davis–Norman)
- Développement en lambda^(1/3) et lambda^(2/3)
- Validation par comparaison numérique
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------- Paramètres -----------------
mu = 0.07
sigma = 0.2
r = 0.02
gamma = 5.0

pi_star = (mu - r) / (gamma * sigma**2)
print(f"π* (Merton) = {pi_star:.4f}")

# ----------------- Asymptotique -----------------
def asymptotic_bandwidth(lam):
    """
    Largeur de la bande d’inaction ~ c * lam^(1/3)
    Constante c = ((3/2) * pi*^2 * (1-pi*)^2 * gamma * sigma^2)^(1/3)
    """
    c = ((1.5) * (pi_star**2) * ((1-pi_star)**2) * gamma * sigma**2)**(1/3)
    delta = c * lam**(1/3)
    b = max(0.0, pi_star - delta)
    s = min(1.0, pi_star + delta)
    return b, s, delta

def asymptotic_value(lam):
    """
    Développement : V ≈ V0 + lam^(2/3) V1
    Ici on approxime V0 par l’utilité frictionless de W=1.
    """
    V0 = 1.0**(1-gamma)/(1-gamma)
    # correction (proportionnelle à -delta^2)
    _, _, delta = asymptotic_bandwidth(lam)
    V1 = -0.5 * gamma * sigma**2 * delta**2
    return V0 + V1

# ----------------- Comparaison numérique -----------------
lams = np.logspace(-4, -1, 10)  # coûts petits
b_list, s_list, V_list = [], [], []

for lam in lams:
    b, s, delta = asymptotic_bandwidth(lam)
    V = asymptotic_value(lam)
    b_list.append(b)
    s_list.append(s)
    V_list.append(V)

# ----------------- Graphiques -----------------
plt.figure(figsize=(12,4))

# 1) Frontières
plt.subplot(1,3,1)
plt.plot(lams, b_list, 'g-o', label="b (asymptotique)")
plt.plot(lams, s_list, 'r-o', label="s (asymptotique)")
plt.xscale("log")
plt.xlabel("λ (log)")
plt.ylabel("Frontières")
plt.title("Frontières asymptotiques b,s")
plt.legend()

# 2) Largeur de bande ~ λ^(1/3)
widths = np.array(s_list) - np.array(b_list)
plt.subplot(1,3,2)
plt.loglog(lams, widths, 'b-o', label="Largeur")
plt.loglog(lams, widths[0]*(lams/lams[0])**(1/3),'k--',label="∝ λ^(1/3)")
plt.xlabel("λ (log)")
plt.ylabel("Largeur")
plt.title("Convergence ~ λ^(1/3)")
plt.legend()

# 3) Valeur
plt.subplot(1,3,3)
plt.plot(lams, V_list, 'm-o', label="V asymptotique")
plt.xlabel("λ")
plt.title("Correction de valeur")
plt.legend()

plt.tight_layout()
plt.show()
