import fire
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve, minimize


def transcendental_eq_minimized(alpha, T, N, B, theta):
    chi_T = 0.5 - B * (1 - theta / T) / theta
    log_arg = 1 - 1 / (alpha ** 3 * N ** (1 / 2))
    if log_arg <= 0:
        return 1e6
    result = (alpha ** 5 - alpha ** 3 + alpha ** 3 * N + alpha ** 6 * N ** (3 / 2) * np.log(log_arg) + chi_T * N ** (
                1 / 2)) ** 2
    return result


def calculate_alpha(T, N, B, theta):
    alpha_initial = 1
    bounds = [(0.16, 1.28)]
    solution = minimize(transcendental_eq_minimized, alpha_initial, args=(T, N, B, theta), bounds=bounds,
                        method='L-BFGS-B')
    if solution.success:
        return solution.x[0]
    else:
        return np.nan

def calculate_error(model_T, model_alpha_squared, exp_T, exp_alpha_squared):
    interp_alpha_squared = np.interp(exp_T, model_T, model_alpha_squared)
    return np.mean((interp_alpha_squared - exp_alpha_squared) ** 2)

def q1_6():
    plt.figure(figsize=(6, 8))
    T_1_C = np.array([16, 17, 18, 19.10, 20, 21.05, 22.10, 25.10, 28, 30, 32.15, 34.10,35.05,37.90,40.10,41,41.5,42.05,43,45.1,50,55.3,59.9,65.15,70])
    T_2_C = np.array([20,21,22,25.10,28,30,32,34,35,36,37,38,40,41,41.50,42,43,45.1,50,55.10,59.9,65.15,70])
    T_1_K = T_1_C + 273.2
    T_2_K = T_2_C + 273.2
    a2_1 = np.array([0.197,0.207,0.213,0.220,0.225,0.221,0.241,0.266,0.338,0.427,0.505,0.588,0.632,0.763,0.853,0.930,1,1.021,1.04,1.055,1.1,1.118,1.169,1.14,1.161])
    a2_2 = np.array([0.186,0.191,0.2,0.22,0.264,0.355,0.433,0.514,0.58,0.656,0.698,0.725,0.836,0.914,1,1.041,1.072,1.14,1.198,1.228,1.257,1.276,1.266])
    plt.scatter(T_1_K,a2_1, label='PMMA-1 Experimental Data', color='orange')
    plt.scatter(T_2_K,a2_2, label='PMMA-2 Experimental Data', color='red')
    spline_1 = UnivariateSpline(T_1_K, a2_1, s=0.02)
    spline_2 = UnivariateSpline(T_2_K, a2_2, s=0.02)
    T_fit_1 = np.linspace(T_1_K.min(), T_1_K.max(), 10000)
    T_fit_2 = np.linspace(T_2_K.min(), T_2_K.max(), 10000)
    a2_fit_1 = spline_1(T_fit_1)
    a2_fit_2 = spline_2(T_fit_2)
    plt.plot(T_fit_1, a2_fit_1, color='orange', linestyle='--')
    plt.plot(T_fit_2, a2_fit_2, color='red', linestyle='--')
    plt.xlabel("Temperature (K)")
    plt.ylabel(r'$\alpha^2$')

    # Model
    B_range = np.arange(20, 130, 10)  # Define the grid for B
    theta = 314.7 # K
    T_values = np.linspace(287, 343, 1000)
    N_values = [25669, 47243]
    results = {}
    lowest_error = 100
    for B in B_range:
        total_error = 0
        model_results = {}
        for N in N_values:
            alpha_squared_values = []
            for T in T_values:
                alpha = calculate_alpha(T, N, B, theta)
                alpha_squared_values.append(alpha ** 2)
            model_results[N] = {'T': T_values, 'alpha_squared': alpha_squared_values}
            total_error += calculate_error(T_values, alpha_squared_values, T_1_K, a2_1)
        if total_error < lowest_error:
            best_B = B
            lowest_error = total_error
            best_results = model_results
    for N, data in best_results.items():
        color = "purple" if N == N_values[0] else "blue"
        plt.plot(data['T'], data['alpha_squared'], label=f"Model Fit for N={N}, B = {best_B}", color=color,
                 linewidth=2.5)
    plt.xlabel('T')
    plt.ylabel(r'$\alpha^2$')
    plt.legend()
    plt.show()
    return


def main():
    q1_6()
    return

if __name__ == "__main__":
    fire.Fire(main())
