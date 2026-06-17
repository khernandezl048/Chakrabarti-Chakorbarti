import numpy as np
import networkx as nx
import time
from joblib import Parallel, delayed

#-----------------------------------FUNCIONES--------------------------------------

def Simulacion(M, edges, prob, Money, T_max, lambdas_com, lamb_inter, community_of):
    m = Money.copy()

    indices  = np.random.choice(len(edges), size=T_max, p=prob)
    epsilons = np.random.uniform(0, 1, size=T_max)

    for t in range(T_max):
        i, j   = edges[indices[t]]
        epsilon = epsilons[t]

        i_com = community_of[i]
        j_com = community_of[j]

        if i_com == j_com:
            lamb = lambdas_com[i_com]
        else:
            # Estrategia 3: promedio entre los lambdas de los dos agentes
            if lamb_inter == "promedio_par":
                lamb = (lambdas_com[i_com] + lambdas_com[j_com]) / 2
            else:
                lamb = lamb_inter  # Estrategia 1 o 2: valor fijo

        delta_m = (1 - lamb) * (epsilon*m[j] - (1-epsilon) * m[i])

        if m[i] + delta_m < 0 or m[j] - delta_m < 0:
            continue

        m[i] += delta_m
        m[j] -= delta_m

    return m

#-----------------------------------RED--------------------------------------

def N_Communities_random(n_com, n_agents, p_out):
    sizes = [int(n_agents)] * n_com
    p_in  = 1 - p_out
    probs = [[p_in if i == j else p_out for j in range(n_com)] for i in range(n_com)]
    G     = nx.stochastic_block_model(sizes, probs, seed=42, directed=False)
    return G

def generar_lambdas(promedio, num, seed=None):
    if not (0 < promedio < 1):
        raise ValueError("El promedio debe estar entre 0 y 1")
    rng = np.random.default_rng(seed)
    suma_objetivo = num * promedio
    while True:
        lambdas = rng.dirichlet(np.ones(num))
        lambdas *= suma_objetivo
        if np.all(lambdas < 1):
            return lambdas

#-----------------------------------PARÁMETROS--------------------------------------

T_max         = int(1e5)
N_simulations = int(1e4)
M             = 5e4
n_com         = 5
n_per         = 100
N             = n_com * n_per

lambdas_com  = generar_lambdas(promedio=0.5, num=n_com)
community_of = np.repeat(np.arange(n_com), n_per)
idx_com      = [np.where(community_of == c)[0] for c in range(n_com)]

G     = N_Communities_random(n_com, n_per, p_out=0.1)
Money = np.full(N, M/N)
edges = np.array(list(G.edges()), dtype=np.int32)
prob  = np.ones(len(edges), dtype=np.float64)
prob /= prob.sum()

# Tres estrategias: (nombre, lamb_inter)
estrategias = [
    ("Promedio",      np.mean(lambdas_com)),   # Estrategia 1
    ("Constante",     0.7),                    # Estrategia 2
    ("Promedio_par",  "promedio_par"),          # Estrategia 3
]

#-----------------------------------SIMULACIÓN--------------------------------------

start_time_total = time.time()

print(f"Comunidades : {n_com}")
print(f"Agentes/com : {n_per} | Total: {N}")
print(f"lambdas_com : {np.round(lambdas_com, 4)}")

for nombre, lamb_inter in estrategias:
    print(f"\n{'='*45}")
    print(f"Estrategia: {nombre}")
    if lamb_inter != "promedio_par":
        print(f"lamb_inter : {lamb_inter:.4f}")
    print(f"{'='*45}")

    start = time.time()

    results = Parallel(n_jobs=8, verbose=10)(
        delayed(Simulacion)(M, edges, prob, Money, T_max, lambdas_com, lamb_inter, community_of)
        for sim in range(N_simulations)
    )

    results = np.array(results)

    # Guardar sistema completo
    np.save(f'Pm_full_{nombre}.npy', results)

    # Guardar por comunidad
    for c in range(n_com):
        np.save(f'Pm_com{c+1}_{nombre}.npy', results[:, idx_com[c]])

    print(f"Estrategia {nombre} completada en {(time.time() - start)/60:.2f} min")

print(f"\n--- Tiempo total: {(time.time() - start_time_total)/60:.2f} minutos ---")