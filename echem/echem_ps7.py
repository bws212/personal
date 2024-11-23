import numpy as np
import matplotlib.pyplot as plt
from sympy import Eq, solve, symbols

def q7_2():
    r = 0.2 # cm
    ohmraw = np.array([0, 100, 400, 900, 1600, 2500, 3600]) # rpm
    ohm = (ohmraw * 2 * np.pi) / 60
    sqrtohm = np.sqrt(ohm)
    I = np.array([0, 1.3, 2.5, 3.8, 5.1, 6.3, 7.6]) # mA
    A = np.pi * r ** 2
    i = I / A
    slope, intercept = np.polyfit(sqrtohm, i, 1)
    fit_line = slope * sqrtohm + intercept
    n = 2
    slim = 1
    F = 96500
    v = 1.002
    C = 50 * 10 ** -3
    D = symbols('D')
    Equation = Eq(slope, 0.62 * ((n * F * C)/slim) * v ** (-1/6) * D ** (2/3))
    sol = solve(Equation, D)
    sol = float(sol[0])
    print(sol)
    plt.figure(figsize=(8,8))
    plt.plot(sqrtohm, fit_line, color='red', alpha=0.5, label=f'Fit (slope = {slope:.2f}), Diffusion = {sol:.2e}')
    plt.scatter(sqrtohm, i, color='black')
    plt.legend()
    plt.xlabel("Sqrt(Ohms) [rad/sec]")
    plt.ylabel("-i limiting [mA/cm^2]")
    plt.title("Levich Plot")
    plt.show()
    return


def main():
    q7_2()
    return

if __name__ == "__main__":
    main()

