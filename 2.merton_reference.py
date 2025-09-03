import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# MERTON (une action risquée + actif sans risque), utility CRRA
# U(x) = x^(1-γ)/(1-γ)  (γ != 1) ; cas limite γ=1 -> U(x)=log x
# ----------------------------

def merton_pi_star(mu, r, sigma, gamma):
    """
    Proportion optimale constante (Merton):
    pi* = (mu - r) / (gamma * sigma^2)
    """
    if sigma <= 0:
        raise ValueError("sigma doit être > 0.")
    if gamma <= 0:
        raise ValueError("gamma doit être > 0.")
    return (mu - r) / (gamma * sigma**2)

def merton_value_function(W0, mu, r, sigma, gamma, T):
    """
    Fonction valeur en t=0 pour utilité CRRA (terminal only, pas de conso intermédiaire).
    - Si gamma != 1:
      V0 = (W0^(1-gamma)/(1-gamma)) * exp( (1-gamma)*[ r + ((mu-r)^2)/(2*gamma*sigma^2) ] * T )
    - Si gamma == 1 (log utility):
      V0 = E[log(W_T)] = log(W0) + [ r + ((mu-r)^2)/(2*sigma^2) ] * T
      (on renvoie la valeur de l'espérance de log(W_T) pour rester cohérent)
    """
    if gamma == 1:
        return np.log(W0) + (r + ((mu - r)**2) / (2 * sigma**2)) * T
    else:
        k = (1 - gamma) * ( r + ((mu - r)**2) / (2 * gamma * sigma**2) )
        return (W0**(1 - gamma) / (1 - gamma)) * np.exp(k * T)

def simulate_wealth_paths(W0, mu, r, sigma, gamma, T, n_steps=252, n_sims=10000, seed=42):
    """
    Simule W_t sous la stratégie constante pi* (exacte) :
      dW_t / W_t = [r + pi*(mu - r)] dt + pi*sigma dB_t
    Discrétisation exacte de la GBM: W_{t+Δ} = W_t * exp((μ_p - 0.5 σ_p^2)Δ + σ_p sqrt(Δ) Z).
    """
    rng = np.random.default_rng(seed)
    pi_star = merton_pi_star(mu, r, sigma, gamma)
    mu_p = r + pi_star * (mu - r)
    sigma_p = abs(pi_star) * sigma  # volatilité du portefeuille

    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    # Incréments gaussiens
    Z = rng.standard_normal(size=(n_sims, n_steps))
    drift = (mu_p - 0.5 * sigma_p**2) * dt
    diffu = sigma_p * np.sqrt(dt)

    # Construction des trajectoires de richesse
    log_increments = drift + diffu * Z
    log_W = np.cumsum(np.column_stack([np.zeros(n_sims), log_increments]), axis=1) + np.log(W0)
    W = np.exp(log_W)

    return t, W, pi_star

def main():
    # ----------------------------
    # Paramètres (modifiez à votre guise)
    # ----------------------------
    W0     = 100.0   # richesse initiale
    mu     = 0.08    # drift de l'actif risqué
    r      = 0.02    # taux sans risque
    sigma  = 0.20    # volatilité de l'actif risqué
    gamma  = 5.0     # aversion au risque relative (CRRA)
    T      = 1.0     # horizon (années)

    n_steps = 252    # pas de temps (jours de bourse ~)
    n_sims  = 20000  # nombre de simulations (pour une belle distribution)

    # ----------------------------
    # Calculs analytiques (pi*, fonction valeur)
    # ----------------------------
    pi_star = merton_pi_star(mu, r, sigma, gamma)

    V0 = merton_value_function(W0, mu, r, sigma, gamma, T)
    # Pour information: espérance/variance de log W_T sous pi*
    mu_p = r + pi_star * (mu - r)
    sigma_p = abs(pi_star) * sigma

    print("=== Merton (Référence frictionless) ===")
    print(f"pi* (proportion optimale en actif risqué) = {pi_star:.6f}")
    if gamma == 1:
        print(f"Fonction valeur (E[log(W_T)]) = {V0:.6f}")
    else:
        print(f"Fonction valeur V0 (CRRA) = {V0:.6e}")
    print(f"μ_portefeuille = {mu_p:.4%}  |  σ_portefeuille = {sigma_p:.4%}  |  Horizon T = {T} an(s)")

    # ----------------------------
    # Simulations
    # ----------------------------
    t, W, pi_star = simulate_wealth_paths(W0, mu, r, sigma, gamma, T, n_steps, n_sims)

    # ----------------------------
    # GRAPHIQUES
    # ----------------------------

    # 1) Trajectoires de richesse (quelques courbes)
    plt.figure(figsize=(8, 4.8))
    n_show = 12
    for i in range(min(n_show, W.shape[0])):
        plt.plot(t, W[i, :], alpha=0.9)
    plt.xlabel("Temps (années)")
    plt.ylabel("Richesse W_t")
    plt.title("Trajectoires simulées de richesse sous π* (Merton)")
    plt.tight_layout()

    # 2) Proportion temporelle (constante ici)
    plt.figure(figsize=(8, 3.5))
    plt.plot(t, np.full_like(t, pi_star))
    plt.xlabel("Temps (années)")
    plt.ylabel("Proportion en actif risqué π_t")
    plt.title("Proportion optimale π* (constante dans Merton)")
    plt.tight_layout()

    # 3) Distribution de W_T (toutes simulations)
    WT = W[:, -1]
    plt.figure(figsize=(8, 4.8))
    plt.hist(WT, bins=60, density=True)
    plt.axvline(np.median(WT), linestyle="--", label=f"Médiane: {np.median(WT):.2f}")
    plt.xlabel("Richesse finale W_T")
    plt.ylabel("Densité")
    plt.title("Distribution de la richesse finale W_T (π* de Merton)")
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
