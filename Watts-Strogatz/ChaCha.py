import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed


#-----------------------------------FUNCIONES--------------------------------------

def Simulacion(M, edges, prob, Money, T_max, lamb):
    m = Money.copy()
    
    indices = np.random.choice(len(edges), size=T_max, p=prob)
    epsilons = np.random.uniform(0, 1, size=T_max)
    
    for t in range(T_max):
        i, j = edges[indices[t]]
        epsilon = epsilons[t]
        
        #delta_m = (1 - lamb) * (m[i] - epsilon * (m[i] + m[j]))
        delta_m = (1 - lamb) * (epsilon*m[j] - (1-epsilon) * (m[i]))
        
        
        if m[i] + delta_m < 0 or m[j] - delta_m < 0:
            continue
        
        m[i] += delta_m
        m[j] -= delta_m
    
    return m


#-----------------------------------SIMULACION--------------------------------------

T_max = int(1e5)
N_simulations = int(1e4)

M = 5e4
N = 500

lamb = 0.1
G = nx.complete_graph(N)
A = nx.to_numpy_array(G)
Money = np.full(N, M/N)

# Precalcular edges y probabilidades una sola vez
'''
edges = np.argwhere(A > 0)
prob = A[edges[:, 0], edges[:, 1]]
prob /= prob.sum()
'''
edges = np.array(list(G.edges()), dtype=np.int32)
prob = np.ones(len(edges), dtype=np.float64)
prob /= prob.sum()
start_time_total = time.time()

results = Parallel(n_jobs=8, verbose=10)(
    delayed(Simulacion)(M, edges, prob, Money, T_max, lamb)
    for sim in range(N_simulations)
)

Money_prom = results

print(f"\n--- Tiempo total: {(time.time() - start_time_total)/60:.2f} minutos ---")

np.save('Pm_full.npy', Money_prom)