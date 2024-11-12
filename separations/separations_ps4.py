import fire
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import symbols, Eq, solve

def q6_21():
    x_b = 0.00041
    P = 103 # kPa
    x_eb = 0.000032
    pi_s_b = 7.85 # kPa
    pi_s_eb = 0.693 # kPa
    L = 0.0475 * 1000 * 1000 * (1/18.02) * 3600 * (1/1000) # kmol/hr
    V = (103000 * 2.41) / (8.314 * (15+273.2)) * 3600 * (1/1000)
    K_b = pi_s_b / (x_b * P)
    K_eb = pi_s_eb / (x_eb * P)
    print(f"K_b: {K_b:.2f}, K_eb: {K_eb:.2f}")
    S_b = (K_b * V) / L
    S_eb = (K_eb * V) / L
    print(f"S_b: {S_b:.2f}, S_eb: {S_eb:.2f}")
    N = np.arange(0,10,1)
    voc_rec = []
    mw_b = 78.11 # g/mol
    mw_eb = 106.167 # g/mol
    in_voc = 29.070 # kg
    mol_b_in = 0.0475 * 1000 * 0.150 * (1/78.11) * 3600 * (1/1000) # kmol/hr
    mol_eb_in = 0.0475 * 1000 * 0.020 * (1/106.167) * 3600 * (1/1000) # kmol/hr
    for n in N:
        frac_b = (S_b ** (n+1) - S_b) / (S_b ** (n+1) - 1)
        frac_eb = (S_eb ** (n+1) - S_eb) / (S_eb ** (n+1) - 1)
        outlet_b = frac_b * mol_b_in
        outlet_eb = frac_eb * mol_eb_in
        outlet_weight_b = outlet_b * mw_b
        outlet_weight_eb = outlet_eb * mw_eb
        combo_weight = outlet_weight_b + outlet_weight_eb
        voc_recovered = (combo_weight / in_voc)
        voc_rec.append(voc_recovered)
        if voc_recovered >= 0.999:
            print(f"number of stages required: {n}, voc recovered: {voc_recovered}")
    plt.plot(N, voc_rec, label="total voc weight % stripped")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Fraction stripped")
    plt.show()
    return


def q7_13():
    plt.figure(figsize=(8,8))
    b = symbols("b")
    equation = Eq(30*0.40, 0.97 * (30-b) + 0.02*b)
    b_sol = float(solve(equation, b)[0])
    print(f"b_sol: {b_sol}")
    z_B_mass = b_sol * 0.02
    t_B_mass = b_sol * 0.98
    z_D_mass = 12 * 0.97
    t_D_mass = 12 * 0.03
    z_B_mols = z_B_mass * 1000 * (1/78.11)
    t_B_mols = t_B_mass * 1000 * (1/92.14)
    z_D_mols = z_D_mass * 1000 * (1/78.11)
    t_D_mols = t_D_mass * 1000 * (1/92.14)
    D_tot = z_D_mols + t_D_mols
    B_tot = z_B_mols + t_B_mols
    z_D_mol_frac = z_D_mols / D_tot
    z_B_mol_frac = z_B_mols / B_tot
    t_D_mol_frac = t_D_mols / D_tot
    t_B_mol_frac = t_B_mols / B_tot
    print(f"Mol frac z in D: {z_D_mol_frac:.4f}, Mol frac z in B: {z_B_mol_frac:.4f}, Mol frac t in D: {t_D_mol_frac:.4f}"
          f"mol fraction t in B: {t_B_mol_frac:.4f}")
    x = np.linspace(0,1,1000)
    y = x
    plt.plot(x,y, label = "45 Degree Line")
    # Q-Line
    y_int = 0.44
    plt.hlines(y_int,0.25, 0.44, color="red", label="q-line")
    # Equilibrium Curve
    y_eq = [0, 0.21, 0.37, 0.51, 0.64, 0.72, 0.79, 0.86, 0.91, 0.96, 0.98, 1]
    x_eq = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    plt.plot(x_eq, y_eq, label = "Equilibrium Curve")
    # Rectifying Section Operating Line
    R = 3.5
    x_RS = np.linspace(0,1,1000)
    y_RS = (R / (R+1)) * x_RS + (1/(R+1)) * z_D_mol_frac
    # Stripping Section OL
    x_int = (y_int - (1 / (R + 1)) * z_D_mol_frac) / (R / (R + 1))
    x_start = 0.0235
    y_start = 0.0235
    plt.plot([x_start, x_int], [y_start, y_int], linestyle="--", color="green",
             label="Stripping Section OL")
    x_RS_limited = x_RS[x_RS >= x_int]
    y_RS_limited = y_RS[x_RS >= x_int]
    plt.plot(x_RS_limited, y_RS_limited, label="RS line", color="orange")
    plt.legend()
    plt.xlabel("mol fraction Benzene in liquid")
    plt.ylabel("mol fraction in Benzene in vapor")
    plt.show()
    return


def q7_16():
    plt.figure(figsize=(8,8))
    # 45 degree  line
    x = np.linspace(0,1,1000)
    y = x
    plt.plot(x,y, color = "green", label = "45 Degree Line")
    # Q-Line
    x_int = 0.50
    plt.vlines(x_int,0.50, 0.708, color="red", label="q-line")
    # Equilibrium Curve
    y_eq = np.array([0, 0.19, 0.37, 0.5, 0.62, 0.71, 0.78, 0.84, 0.9, 0.96, 1])
    x_eq = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(x_eq, y_eq, label = "Equilibrium Curve")
    # Rectifying Section OL
    R = 3
    x_RS = np.linspace(0,1,1000)
    x_D = 0.90
    y_RS = (R / (R+1)) * x_RS + (1/(R+1)) * x_D
    y_int = 0.50
    x_intercept = (y_int - (1 / (R + 1)) * x_D) / (R / (R + 1))
    x_RS_limited = x_RS[x_RS >= x_int]
    y_RS_limited = y_RS[x_RS >= x_int]
    plt.plot(x_RS_limited, y_RS_limited, label="RS line", color="orange")

    # SS OL
    x_start_2 = 0.075
    y_start_2 = 0.075
    y_int_for_SS = (R / (R + 1)) * x_int + (1 / (R + 1)) * x_D
    slope = (y_int_for_SS - y_start_2) / (x_int - x_start_2)
    plt.plot([x_start_2, x_int], [y_start_2, y_int_for_SS], color="blue", label=f"SS OL, slope = {slope:.3f}")
    plt.legend()
    plt.xlabel("mol fraction A in liquid")
    plt.ylabel("mol fraction A in vapor")
    plt.show()
    b = symbols("b")
    equation = Eq(100 * 0.50, 0.90 * (100 - b) + 0.075 * b)
    b_sol = float(solve(equation, b)[0])
    print(f"b_sol: {b_sol}")
    return


def main():
    # q6_21()
    # q7_13()
    q7_16()
    return

if __name__ == "__main__":
    fire.Fire(main())