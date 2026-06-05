import numpy as np
import time

# -----------------------------------FUNCIONES--------------------------------------

def Simulacion_BM(N, W, J, eta_mean, sigma, T_max):
    omega = np.full(N, W / N, dtype=np.float64)

    agentes = np.random.randint(0, N, size=T_max)
    etas    = np.random.normal(loc=eta_mean, scale=np.sqrt(2) * sigma, size=T_max)

    for t in range(T_max):
        i   = agentes[t]
        eta = etas[t]

        omega_mean  = np.mean(omega)
        omega_tilde = omega / omega_mean

        w_i = omega_tilde[i]

        delta = (eta - eta_mean - sigma**2) * w_i + J * (1 - w_i)

        if w_i + delta < 0:
            continue

        omega_tilde[i] += delta
        omega = omega_tilde * omega_mean

    return omega / np.mean(omega)


# -----------------------------------PARÁMETROS--------------------------------------

N        = 1000
W        = 1000
eta_mean = 0.0
sigma    = 1.0
T_max    = int(4e4)
N_simulations = int(1e4)

Js = np.array([1.0, 2.0, 3.0])

# -----------------------------------SIMULACIÓN--------------------------------------

start_time_total = time.time()

for J in Js:
    print(f"\n{'='*40}")
    print(f"Simulando J = {J}")
    print(f"{'='*40}")

    start = time.time()

    results = []
    for sim in range(N_simulations):
        results.append(Simulacion_BM(N, W, J, eta_mean, sigma, T_max))
        if (sim + 1) % 100 == 0:
            print(f"  Simulacion {sim+1}/{N_simulations}")

    np.save(f'Pm_BM_J{J:.1f}.npy', results)

    print(f"J={J:.1f} completado en {(time.time() - start)/60:.2f} min")

print(f"\n--- Tiempo total: {(time.time() - start_time_total)/60:.2f} minutos ---")