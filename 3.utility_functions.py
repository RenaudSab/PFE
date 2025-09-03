import numpy as np
import matplotlib.pyplot as plt

W0 = 100.0                # richesse de référence
w_grid = np.linspace(1e-3, 400, 1000)  # grille de richesse positive

gamma_crra = 0.5          # CRRA (aversion relative constante R = gamma)
alpha_cara  = 0.02        # CARA (aversion absolue constante A = alpha)
b_quad      = 0.002       # Quadratique U(c)=c - (b/2)c^2, domaine c < 1/b
# (assure que max(w) < 1/b pour éviter U'(c)<=0)
assert w_grid.max() < 1.0 / b_quad, "Augmente 1/b ou réduis la grille."

mu, r, sigma = 0.08, 0.02, 0.20

def U_crra(c, gamma):
    if np.isclose(gamma, 1.0):
        return np.log(c)
    return (c**(1-gamma) - 1) / (1 - gamma)

def U_log(c):
    return np.log(c)

def U_cara(c, alpha):
    return -np.exp(-alpha*c) / alpha

def U_quad(c, b):
    return c - 0.5*b*c**2  # valable tant que c < 1/b (U'>0)

def A_crra(c, gamma):  # aversion absolue
    return gamma / c

def R_crra(c, gamma):  # aversion relative
    return gamma * np.ones_like(c)

def A_log(c):          # cas gamma=1
    return 1.0 / c

def R_log(c):
    return np.ones_like(c)

def A_cara(c, alpha):
    return alpha * np.ones_like(c)

def R_cara(c, alpha):
    return alpha * c

def A_quad(c, b):
    # U'(c)=1 - b c, U''(c)=-b  => A(c)= b/(1-bc), croissante (IARA)
    return b / (1.0 - b*c)

def R_quad(c, b):
    return (b*c) / (1.0 - b*c)

gamma_mv = 5.0  # aversion MV (Markowitz)

pi_crra = (mu - r) / (gamma_crra * sigma**2)
pi_log  = (mu - r) / (1.0        * sigma**2)
y_cara  = (mu - r) / (alpha_cara * sigma**2)
w_cara_vs_W = y_cara / w_grid
w_mv    = (mu - r) / (gamma_mv   * sigma**2)


def norm(u):  # centre à U(1)=0
    return u - u[np.argmin(np.abs(w_grid - 1.0))]

plt.figure(figsize=(8.5, 5))
plt.plot(w_grid, norm(U_crra(w_grid, gamma_crra)), label=f"CRRA (γ={gamma_crra})")
plt.plot(w_grid, norm(U_log(w_grid)),               label="Log (γ=1)")
plt.plot(w_grid, norm(U_cara(w_grid, alpha_cara)),  label=f"CARA (α={alpha_cara})")
plt.plot(w_grid, norm(U_quad(w_grid, b_quad)),      label=f"Quadratique (b={b_quad})")
plt.xlabel("Richesse / consommation c")
plt.ylabel("Utilité (centrée à U(1)=0)")
plt.title("Formes d'utilité (Chapitre 4)")
plt.legend()
plt.tight_layout()

#    1) Aversion absolue A(c)
plt.figure(figsize=(8.5, 4.2))
plt.plot(w_grid, A_crra(w_grid, gamma_crra), label=f"A_CRRA=γ/c (γ={gamma_crra})")
plt.plot(w_grid, A_log(w_grid),              label="A_Log=1/c")
plt.plot(w_grid, A_cara(w_grid, alpha_cara), label=f"A_CARA=α (α={alpha_cara})")
plt.plot(w_grid, A_quad(w_grid, b_quad),     label=f"A_Quad=b/(1-bc) (b={b_quad})")
plt.xlabel("c")
plt.ylabel("A(c)")
plt.title("Aversion absolue au risque A(c)")
plt.ylim(0, np.percentile(A_quad(w_grid, b_quad), 99))  # coupe l'extrême
plt.legend()
plt.tight_layout()

#    2) Aversion relative R(c)
plt.figure(figsize=(8.5, 4.2))
plt.plot(w_grid, R_crra(w_grid, gamma_crra), label=f"R_CRRA=γ")
plt.plot(w_grid, R_log(w_grid),              label="R_Log=1")
plt.plot(w_grid, R_cara(w_grid, alpha_cara), label="R_CARA=α c")
plt.plot(w_grid, R_quad(w_grid, b_quad),     label="R_Quad=(bc)/(1-bc)")
plt.xlabel("c")
plt.ylabel("R(c)")
plt.title("Aversion relative au risque R(c)")
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8.5, 4.8))
plt.plot(w_grid, np.full_like(w_grid, pi_crra), label=f"CRRA π*={(pi_crra):.2f}")
plt.plot(w_grid, np.full_like(w_grid, pi_log),  label=f"Log π*={(pi_log):.2f}")
plt.plot(w_grid, w_cara_vs_W,                  label="CARA w*(W)= y*/W (décroît)")
plt.plot(w_grid, np.full_like(w_grid, w_mv),    label=f"Quadratique/MV w*={(w_mv):.2f}")
plt.xlabel("Richesse W")
plt.ylabel("Fraction optimale en actif risqué")
plt.title("Décisions optimales : comparaison par utilité")
plt.ylim(0, min(1.2, max(1.1*np.max(w_cara_vs_W), pi_crra, pi_log, w_mv)+0.1))
plt.legend()
plt.tight_layout()

plt.show()
