import numpy as np
import fire
import matplotlib.pyplot as plt

# Global Vars
w = 500 # um
h = 100 # um
Q = np.arange(10, 30, 2)
Q_r = Q * 2.77778 * 10 ** 8 # um^3 / sec

def time_eqn(h_s, V_m = 7, s_lim = 1, D_lim = 10**-5, c_inf = 1*10**-4, F=10**5, d=0.01):
    time = ((V_m * s_lim * D_lim * c_inf) / F ) * (h_s/(d **2))
    return time


def U_eqn(Q):
    U = Q / (2 * w * h)
    return U

def diff_eqn(U, x):
    D_lim = 10 ** 3  # um^2 / sec
    diff = (U / (3 * h * D_lim * x)) ** (-1/3)
    return diff

def q6_4():
    plt.figure(figsize=(10, 6))
    diff_val = []
    for q in Q_r:
        x = np.linspace(0.0000001,1000,10000)
        U = U_eqn(q)
        diff = diff_eqn(U, x)
        plt.plot(x, diff, label=f'Q = {q :.2f} um^3/sec')
        idx = (np.abs(x - 100)).argmin()
        diff_val.append(diff[idx])
    # Label the plot
    plt.xlabel('Position (um)')
    plt.ylabel('Diffusion Layer (um)')
    plt.legend()
    plt.grid(True)
    plt.show()
    # For L = 100, x = 100
    if diff_val:  # Ensure diff_val is not empty
        avg_diff = np.mean(diff_val)
    else:
        avg_diff = np.nan  # Return NaN if no values are found
    print(avg_diff)
    return

def q6_8():
    h_s0 = 0.001
    h_pol = np.linspace(0.0001, 0.001, 10000)
    h_plate = np.linspace(0.001, 0.01, 10000)
    pol_time = np.array([time_eqn(h) for h in h_pol])
    plate_time = np.array([time_eqn(h) for h in h_plate])
    plt.plot(pol_time, h_pol)
    plt.plot(plate_time, h_plate)
    plt.show()
    return

def main():
    # q6_4()
    q6_8()
    return

if __name__ == "__main__":
    fire.Fire(main())