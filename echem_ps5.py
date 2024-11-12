import fire
import matplotlib.pyplot as plt
import numpy as np


def q5_8():
    i_0 = 0.1
    alpha = 0.5
    F = 96485
    R = 8.314
    T = 298
    kappa = 0.1
    L = 1.0

    def tafel_eq(V, i_0, alpha, F, R, T, L, kappa):
        i_avg = i_0 * np.exp((-alpha * F * (V - (i_0 * L / kappa))) / (R * T))

        return i_avg

    V_values = np.linspace(0.1, 0.8, 100)
    i_avg_values = [tafel_eq(V, i_0, alpha, F, R, T, L, kappa) for V in V_values]
    print(np.mean(i_avg_values))
    R_ct_values = [R * T / (alpha * F * abs(i_avg)) for i_avg in i_avg_values]
    R_ohm = L / kappa
    resistance_ratios = [R_ct / R_ohm for R_ct in R_ct_values]
    plt.plot(resistance_ratios, V_values)
    plt.xlabel('R_ct / R_ohm')
    plt.ylabel('Applied Potential (V)')
    plt.title('V vs Ratio of Charge Transfer to Ohmic Resistance')
    plt.show()
    return



def q5_10():
    alpha_c = 0.5
    i_0 = 1 * 10 ** -6  # Exchange current density (A/cm²)
    F = 96500  # Faraday's constant (C/mol)
    C_inf = np.linspace(1, 5, 3)  # Concentration range (mol/cm³)
    D = 10 ** -5  # Diffusion coefficient (cm²/s)
    R = 8.314  # Universal gas constant (J/(mol·K))
    T = 298  # Temperature (K)
    delta = np.linspace(0.0001, 0.001, 3)  # Diffusion layer thickness range (cm)
    surf = np.linspace(0, 0.4, 10000)  # Overpotential range (V)

    # Create arrays to store the limiting currents and total currents
    for c in C_inf:  # Loop over concentration values
        for d in delta:  # Loop over diffusion layer thickness values
            i_lim = (F * D * c) / d  # Limiting current (A/cm²)
            i = i_0 * (1 - (i_0 / i_lim)) * np.exp(-alpha_c * F * surf / (R * T))  # Current (A/cm²)
            plt.plot(surf, i, label=f'C_inf={c}, delta={d}')

    plt.xlabel('Overpotential (V)')
    plt.ylabel('Current density (A/cm²)')
    plt.legend()
    plt.title('Current vs Overpotential with varying C_inf and delta')
    plt.show()
    return





def main():
    q5_10()
    q5_8()
    return

if __name__ == "__main__":
    fire.Fire(main())
