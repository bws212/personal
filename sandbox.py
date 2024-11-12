import matplotlib.pyplot as plt
import fire
import numpy as np
from scipy.optimize import least_squares

T = [81.51, 80.66, 80.28, 80.21, 80.16, 80.32, 81.25, 81.64, 83.85, 84.02, 93.00, 100.00, 82.5]
lmolpercI = [94.42, 85.67, 76.93, 68.10, 67.94, 60.30, 34.96, 28.68, 9.10, 8.41, 1.18, 0, 100]
VmolpercI = [91.60, 82.70, 74.21, 68.26, 68.21, 64.22, 55.16, 53.44, 47.06, 46.20, 21.95, 0, 100]


def q4():
    wtA_water_rich = [1.41, 2.89, 6.42, 13.30, 25.50, 36.70, 45.30, 46.40]  # Weight % Acetic Acid (A)
    wtW_water_rich = [97.1, 95.5, 91.7, 84.4, 71.1, 58.9, 45.1, 37.1]  # Weight % Water (W)
    wtE_water_rich = [1.49, 1.61, 1.88, 2.3, 3.4, 4.4, 9.6, 16.5]  # Weight % Ether (E)

    wtA_ether_rich = [0.37, 0.79, 1.93, 4.82, 11.4, 21.6, 31.1, 36.2]  # Weight % Acetic Acid (A)
    wtW_ether_rich = [0.73, 0.81, 0.97, 1.88, 3.9, 6.9, 10.8, 15.1]  # Weight % Water (W)
    wtE_ether_rich = [98.9, 98.4, 97.1, 93.3, 84.7, 71.5, 58.1, 48.7]

    plt.figure(figsize=(10, 6))

    # Plotting Water (wt% W) vs Acetic Acid (wt% A) for the water-rich layer
    plt.plot(wtW_water_rich, wtA_water_rich, label='Water-Rich Layer', marker='o', linestyle='-', color='blue')

    # Plotting Water (wt% W) vs Acetic Acid (wt% A) for the ether-rich layer
    plt.plot(wtW_ether_rich, wtA_ether_rich, label='Ether-Rich Layer', marker='x', linestyle='-', color='red')

    # Plotting the 45-degree line from (0, 100) to (100, 0)
    plt.plot([0, 100], [100, 0], linestyle='--', color='gray', label='45° Line (x=0, y=100 to x=100, y=0)')

    # Adding labels, title, and legend
    plt.xlabel('Wt% Water (W)')
    plt.ylabel('Wt% Acetic Acid (A)')
    plt.title('LLE Separation Graph: Wt% Water vs Wt% Acetic Acid with 45° Line')
    plt.legend()
    plt.grid(True)
    plt.show()
    return






def residuals_safe(params, T_exp, P_exp):
    A, B, C = params
    # Ensure T + C remains positive and above a small threshold (e.g., 1) to avoid division issues
    T_adjusted = T_exp + np.abs(C) + 1
    P_pred = 10 ** (A - B / T_adjusted)
    return np.log10(P_pred) - np.log10(P_exp)


def one_e():
    P_exp = np.array([200, 400, 760])  # Vapor pressures in torr
    T_iso_exp = np.array([53, 67.8, 82.5])  # Temperatures for isopropanol in Celsius
    T_water_exp = np.array([66.5, 83, 100])
    # Updated initial guesses
    initial_guess_iso_safe = [8, 2000, -50]
    initial_guess_water_safe = [8, 2000, -50]

    result_iso_safe = least_squares(residuals_safe, initial_guess_iso_safe, args=(T_iso_exp, P_exp), max_nfev=10000)
    Ai, Bi, Ci = result_iso_safe.x

    result_water_safe = least_squares(residuals_safe, initial_guess_water_safe, args=(T_water_exp, P_exp),
                                      max_nfev=10000)
    Aw, Bw, Cw = result_water_safe.x

    # Display the fitted Antoine constants using the refined method
    print(Ai, Bi, Ci, Aw, Bw, Cw)
    T_range = np.linspace(82.5, 100, 100)
    Pw = 10 ** (Aw - (Bw/(T_range + (-1*Cw))))
    Pi = 10 ** (Ai - (Bi/(T_range + (-1*Ci))))
    Kw = Pw / 760
    Ki = Pi / 760
    xi = (1-Kw) / (Ki-Kw)
    yi = Ki * xi

    plt.figure(figsize=(10, 6))
    plt.plot(xi, T_range, label='Liquid phase (T-x)', marker='o')
    plt.plot(yi, T_range, label='Vapor phase (T-y)', marker='x')

    plt.axhline(y=89, color='pink', linestyle='-', label='T = 89°C')
    plt.xlabel('Mol percent Isopropanol (%)')
    plt.ylabel('Temperature (°C)')
    plt.title('T-x-y Phase Diagram (Updated)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(xi, yi, label='x-y equilibrium curve', marker='o')
    slope = -0.33
    x_feed = 0.40
    y_feed = 0.40
    x_q_line = [0, x_feed, 0.40]
    y_q_line = [y_feed + (slope * (0 - x_feed)), y_feed, y_feed + (slope * (0.40 - x_feed))]
    plt.plot(x_q_line, y_q_line, label='q-line (slope = -0.33)', linestyle='-', color='red')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='y = x (45° line)')
    plt.xlabel('Mole Fraction of Isopropanol in Liquid phase (x)')
    plt.ylabel('Mole Fraction of Isopropanol in Vapor phase (y)')
    plt.title('x-y Phase Diagram using K-values')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


def txydiagrams():
    sorted_indices = sorted(range(len(T)), key=lambda k: lmolpercI[k])
    T_sorted = [T[i] for i in sorted_indices]
    lmolpercI_sorted = [lmolpercI[i] for i in sorted_indices]
    VmolpercI_sorted = [VmolpercI[i] for i in sorted_indices]

    # Plotting T-x-y diagram with sorted data
    plt.figure(figsize=(10, 6))
    plt.plot(lmolpercI_sorted, T_sorted, label='Liquid phase (T-x)', marker='o')
    plt.plot(VmolpercI_sorted, T_sorted, label='Vapor phase (T-y)', marker='x')
    plt.axhline(y=89, color='pink', linestyle='-', label='T = 89°C')
    plt.xlabel('Mol percent Isopropanol (%)')
    plt.ylabel('Temperature (°C)')
    plt.title('T-x-y Phase Diagram (Updated)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting x-y diagram with sorted data
    plt.figure(figsize=(10, 6))
    plt.plot(lmolpercI_sorted, VmolpercI_sorted, label='x-y equilibrium curve', marker='o')
    plt.plot([0, 100], [0, 100], linestyle='--', color='gray', label='y=x (45° line)')
    x_feed = 40
    y_feed = 40

    # Generating points for the q-line
    slope = -1 / 3
    x_q_line = [0, x_feed, 40]
    y_q_line = [y_feed + (slope * (0 - x_feed)), y_feed, y_feed + (slope * (40 - x_feed))]
    plt.plot(x_q_line, y_q_line, label='q-line (slope = -0.33)', linestyle='-', color='red')
    plt.xlabel('Mol percent Isopropanol in Liquid phase (x)')
    plt.ylabel('Mol percent Isopropanol in Vapor phase (y)')
    plt.title('x-y Phase Diagram (Updated)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


def main():
    txydiagrams()
    one_e()
    q4()

if __name__ == "__main__":
    fire.Fire(main())
