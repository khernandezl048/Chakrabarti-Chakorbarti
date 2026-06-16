import numpy as np
import networkx as nx
import time
from joblib import Parallel, delayed

#-----------------------------------FUNCIONES--------------------------------------

def Simulacion(M, edges, prob, Money, T_max, lambdas_com, lamb_inter, community_of):
    m = Money.copy()
    
    indices = np.random.choice(len(edges), size=T_max, p=prob)
    epsilons = np.random.uniform(0, 1, size=T_max)
    
    for t in range(T_max):
        i, j = edges[indices[t]]
        epsilon = epsilons[t]

        i_com = community_of[i]
        j_com = community_of[j]

        lamb = lambdas_com[i_com] if i_com == j_com else lamb_inter

        delta_m = (1 - lamb) * (epsilon*m[j] - (1-epsilon) * m[i])
        
        if m[i] + delta_m < 0 or m[j] - delta_m < 0:
            continue
        
        m[i] += delta_m
        m[j] -= delta_m
    
    return m

#-------------------------------------------RED CON N COMUNIDADES-------------------------------------------

def N_Communities_random(n_com, n_agents, p_out):
    """
    n_com     : número de comunidades
    n_agents  : agentes por comunidad
    p_out     : probabilidad de conexión entre comunidades distintas
    """
    sizes = [int(n_agents)] * n_com
    p_in  = 1 - p_out

    # Matriz de probabilidades: p_in en diagonal, p_out fuera
    probs = [[p_in if i == j else p_out for j in range(n_com)] for i in range(n_com)]

    G = nx.stochastic_block_model(sizes, probs, seed=42, directed=False)
    return G


#-----------------------------------GENERADOR DE LAMBDAS--------------------------------------
def generar_lambdas(promedio, num, seed=None):
    """
    Genera num valores en (0,1) con promedio exactamente igual
    a 'promedio'.
    """

    if not (0 < promedio < 1):
        raise ValueError("El promedio debe estar entre 0 y 1")

    rng = np.random.default_rng(seed)

    suma_objetivo = num * promedio

    while True:
        # Genera números positivos que suman 1
        lambdas = rng.dirichlet(np.ones(num))
        
        # Escala a la suma deseada
        lambdas *= suma_objetivo
        
        # Acepta solo si todos son <1
        if np.all(lambdas < 1):
            return lambdas

#-----------------------------------SIMULACION--------------------------------------

T_max        = int(1e5)
N_simulations = int(1e3)

M     = 5e4
n_com = 10           # <-- número de comunidades
n_per = 50         # <-- agentes por comunidad  (N = n_com * n_per)
N     = n_com * n_per

# Lambda de cada comunidad (lista de longitud n_com)
lambdas_com = generar_lambdas(promedio=0.5, num=n_com)         # un valor por comunidad
lamb_inter   = np.mean(lambdas_com)           # entre comunidades distintas
#lamb_inter   = 0.7

assert len(lambdas_com) == n_com, "lambdas_com debe tener un valor por comunidad"

# Vector que indica a qué comunidad pertenece cada agente
community_of = np.repeat(np.arange(n_com), n_per)   # [0,0,...,1,1,...,n_com-1,...]

# Índices de cada comunidad
idx_com = [np.where(community_of == c)[0] for c in range(n_com)]

G = N_Communities_random(n_com, n_per, p_out=0.1)
Money = np.full(N, M/N)

edges = np.array(list(G.edges()), dtype=np.int32)
prob  = np.ones(len(edges), dtype=np.float64)
prob /= prob.sum()

start_time_total = time.time()

print(f"Comunidades : {n_com}")
print(f"Agentes/com : {n_per}  |  Total: {N}")
print(f"lambdas_com : {lambdas_com}")
print(f"lamb_inter  : {lamb_inter:.4f}")

results = Parallel(n_jobs=8, verbose=10)(
    delayed(Simulacion)(M, edges, prob, Money, T_max, lambdas_com, lamb_inter, community_of)
    for sim in range(N_simulations)
)

results = np.array(results)  # shape (N_simulations, N)

# Guardar sistema completo
np.save('Pm_full.npy', results)

# Guardar cada comunidad por separado
for c in range(n_com):
    np.save(f'Pm_com{c+1}.npy', results[:, idx_com[c]])

print(f"\n--- Tiempo total: {(time.time() - start_time_total)/60:.2f} minutos ---")