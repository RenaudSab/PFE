"""
Multi-Asset QVI — version avec cône de solvabilité rempli

- CRRA avec coûts proportionnels (Kabanov)
- QVI discrète (grille du simplexe)
- KD-tree pour nearest neighbor
- Graphiques :
    * Région d’inaction vs trade (2D)
    * Cône de solvabilité K (3D avec faces convexes)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ------------------ Paramètres ------------------
T = 1.0
N = 24
dt = T / N
gamma = 15.0

m = 2                       # actifs risqués
d = m + 1                   # total (incl. sans risque)
r = 0.02
mu = np.array([0.07, 0.09])
sigma = np.array([0.20, 0.25])
rho = np.array([[1.0, 0.3],[0.3, 1.0]])

lambda_vec = np.array([0.0, 0.003, 0.004])   # spreads
Lambda = lambda_vec[:, None] + lambda_vec[None, :]

n_mc = 400
rng = np.random.default_rng(123)

# ------------------ CRRA ------------------
def u_crra(w):
    w = np.maximum(w, 1e-12)
    return np.log(w) if gamma == 1 else w**(1.0-gamma)/(1.0-gamma)

# ------------------ Tirages ------------------
L = np.linalg.cholesky(rho)
Z = rng.standard_normal((N, n_mc, m))
Z_corr = Z @ L.T
S_rel = np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z_corr)
B_rel = float(np.exp(r*dt))

# ------------------ Grilles ------------------
def simplex_grid_d3(M=10, eps=0.02):
    pts=[]
    for i in range(M+1):
        for j in range(M+1-i):
            k=M-i-j
            w=np.array([i/M,j/M,k/M])
            w=(1-2*eps)*w+eps*np.array([1,1,1])
            pts.append(w)
    return np.array(pts)

Wgrid = simplex_grid_d3(M=12, eps=0.02)   # états
Cgrid = simplex_grid_d3(M=6, eps=0.02)    # contrôles
n_states, n_controls = len(Wgrid), len(Cgrid)

tree = cKDTree(Wgrid)

# ------------------ Fonctions ------------------
def trade_cost(w_from, w_to):
    return float(np.dot(lambda_vec, np.abs(w_to-w_from)))

def step_values_all(w_from, Cgrid, V_next, t):
    """Valeurs pour tous les contrôles"""
    R_risk=S_rel[t]
    R=np.concatenate([np.full((n_mc,1),B_rel),R_risk],axis=1)
    vals=[]
    for c in Cgrid:
        tau=trade_cost(w_from,c)
        if tau>=1: vals.append(-np.inf); continue
        M=R@c; M=np.maximum(M,1e-16)
        w_next=(R*c)/M[:,None]
        idx=tree.query(w_next)[1]
        Vn=V_next[idx]
        factor=(1-tau)**(1-gamma)*M**(1-gamma)
        vals.append(np.mean(factor*Vn))
    return np.array(vals)

# ------------------ QVI backward ------------------
V=np.zeros((N+1,n_states))
V[N,:]=u_crra(1.0)

print(f"QVI: états={n_states}, contrôles={n_controls}, N={N}")
for t in range(N-1,-1,-1):
    V_next=V[t+1,:]; V_t=np.empty(n_states)
    for i,w in enumerate(Wgrid):
        vals=step_values_all(w,Cgrid,V_next,t)
        V_t[i]=np.max(vals)
    V[t,:]=V_t
    if (N-t)%4==0: print(f"t={t:02d} | V min={V_t.min():.2e}, max={V_t.max():.2e}")

print("QVI terminée.")

# ------------------ Bande d'inaction t=0 ------------------
V1=V[1,:]; inaction=np.zeros(n_states,bool)
for i,w in enumerate(Wgrid):
    vals=step_values_all(w,Cgrid,V1,0)
    # valeur inaction = contrôle le plus proche de w
    j_close=np.argmin(np.linalg.norm(Cgrid-w,axis=1))
    v_nt=vals[j_close]
    v_best=vals.max()
    inaction[i]=(v_nt>=v_best-1e-12)
print(f"Inaction: {inaction.mean()*100:.1f}% des états")

# ------------------ Graphiques ------------------
def bary_to_cart_2d(w):
    A=np.array([[0,0],[1,0],[0.5,np.sqrt(3)/2]])
    return w@A

# Région d'action/inaction
xy=np.array([bary_to_cart_2d(w) for w in Wgrid])
plt.figure(figsize=(7,6))
plt.scatter(xy[inaction,0],xy[inaction,1],c='g',s=25,label="Inaction")
plt.scatter(xy[~inaction,0],xy[~inaction,1],c='r',s=25,label="Trade")
plt.title("Région d’action vs inaction (t=0)")
plt.axis('equal');plt.axis('off');plt.legend()

# Cône de solvabilité K (3D avec faces convexes)
fig=plt.figure(figsize=(7,6))
ax=fig.add_subplot(111,projection='3d')
e=np.eye(3)
gens=[]
for i in range(3):
    for j in range(3):
        if i==j: continue
        qij=(1+lambda_vec[i])/max(1e-12,1-lambda_vec[j])
        g=e[i]-qij*e[j]
        gens.append(g)

gens=np.array(gens)
hull=ConvexHull(gens)
for simplex in hull.simplices:
    tri=gens[simplex]
    poly=Poly3DCollection([tri],alpha=0.3,facecolor="cyan")
    ax.add_collection3d(poly)
ax.scatter(gens[:,0],gens[:,1],gens[:,2],c='k')
ax.set_title("Cône de solvabilité K (faces convexes)")
ax.set_xlabel("asset 0");ax.set_ylabel("asset 1");ax.set_zlabel("asset 2")

plt.show()
