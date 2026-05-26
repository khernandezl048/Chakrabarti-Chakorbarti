import numpy as np
import networkx as nx
import time
from joblib import Parallel, delayed

#-----------------------------------FUNCIONES--------------------------------------

def Simulacion(M, edges, prob, Money, T_max, lamb):
    m = Money.copy()
    
    indices = np.random.choice(len(edges), size=T_max, p=prob)
    epsilons = np.random.uniform(0, 1, size=T_max)
    
    for t in range(T_max):
        i, j = edges[indices[t]]
        epsilon = epsilons[t]
        
        delta_m = (1 - lamb) * (epsilon*m[j] - (1-epsilon) * m[i])
        
        if m[i] + delta_m < 0 or m[j] - delta_m < 0:
            continue
        
        m[i] += delta_m
        m[j] -= delta_m
    
    return m

#-------------------------------------------RED CON DOS COMUNIDADES-------------------------------------------

def Two_Communities_random(n, p):
    sizes = [int(n), int(n)]
    p_in = 1 - p
    p_out = p
    probs = [[p_in, p_out], [p_out, p_in]]
    G = nx.stochastic_block_model(sizes, probs, seed=42, directed=True)
    return G

#-----------------------------------SIMULACION--------------------------------------

T_max = int(1e5)
N_simulations = int(1e3)

M = 5e4
N = 500
n_com = N // 2  # tamaño de cada comunidad

# Índices de cada comunidad
idx_com1 = np.arange(0, n_com)          # agentes 0   a 249
idx_com2 = np.arange(n_com, N)          # agentes 250 a 499

G = Two_Communities_random(n_com, 0.1)
Money = np.full(N, M/N)

edges = np.array(list(G.edges()), dtype=np.int32)
prob = np.ones(len(edges), dtype=np.float64)
prob /= prob.sum()

lambdas = np.round(np.arange(0.0, 1.0, 0.1), 1)

start_time_total = time.time()

for lamb in lambdas:
    print(f"\n{'='*40}")
    print(f"Simulando lamb = {lamb}")
    print(f"{'='*40}")
    
    start = time.time()
    
    results = Parallel(n_jobs=8, verbose=10)(
        delayed(Simulacion)(M, edges, prob, Money, T_max, lamb)
        for sim in range(N_simulations)
    )

    results = np.array(results)  # shape (N_simulations, N)

    # Sistema completo
    np.save(f'Pm_lamb{lamb:.1f}.npy', results)

    # Comunidad 1 (agentes 0 a 249)
    np.save(f'Pm_com1_lamb{lamb:.1f}.npy', results[:, idx_com1])

    # Comunidad 2 (agentes 250 a 499)
    np.save(f'Pm_com2_lamb{lamb:.1f}.npy', results[:, idx_com2])
    
    print(f"lamb={lamb:.1f} completado en {(time.time() - start)/60:.2f} min")

print(f"\n--- Tiempo total: {(time.time() - start_time_total)/60:.2f} minutos ---")