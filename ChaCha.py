import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numba import njit
from scipy.stats import gaussian_kde
from scipy.linalg import eigvals
from scipy.linalg import expm
from matplotlib import rc


#-----------------------------------FUNCIONES--------------------------------------

def Simulacion(M, A, Money, T_max, lamb):
    m = Money.copy()
    
    # Precalcular edges y probabilidades una sola vez
    edges = np.argwhere(A > 0)
    A_values = A[edges[:, 0], edges[:, 1]]
    prob = A_values / A_values.sum()
    
    # Pregenerar todos los números aleatorios de una vez
    indices = np.random.choice(len(edges), size=T_max, p=prob)
    epsilons = np.random.uniform(0, 1, size=T_max)
    
    for t in range(T_max):
        i, j = edges[indices[t]]
        epsilon = epsilons[t]
        
        delta_m = (1 - lamb) * (m[i] - epsilon * (m[i] + m[j]))
        
        if m[i] + delta_m < 0 or m[j] - delta_m < 0:
            continue
        
        m[i] += delta_m
        m[j] -= delta_m
    
    return m


#-----------------------------------SIMULACION--------------------------------------

T_max = int(1e6)
N_simulations = 100

M = 500
N = 500

lamb = 0.0
G =  nx.complete_graph(N) 
A = nx.to_numpy_array(G)
Money = np.full(N, M/N)
Money_prom = np.zeros(len(Money))

start_time_total = time.time()
for sim in range(N_simulations):
    Money_prom += Simulacion(M, A, Money, T_max, lamb)
    print(f"\n  Simulación {sim+1}/{N_simulations} lista.")
    print(f"--- Tiempo parcial: {(time.time() - start_time_total)/60:.2f} minutos ---")

print(f"\n--- Tiempo total: {(time.time() - start_time_total)/60:.2f} minutos ---")

Money_prom /= N_simulations
np.save('Pm_full.npy',Money_prom)