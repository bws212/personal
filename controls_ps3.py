import fire
import numpy as np
import control.matlab as control
from sympy import symbols, expand, Eq, solve, sqrt, pi, ln
import matplotlib.pyplot as plt


def q1_a_to_c():
    s = symbols('s')
    poly1 = (4*s+1)
    poly2 = (s+1)
    product = expand(poly1*poly2)
    print(product)
    K = symbols('K')
    zeta = (5/4) * (1 / sqrt(1+K))
    eqn = Eq(zeta, 0.00000001)
    sol1 = solve(eqn, K)
    print(sol1)
    eqn2 = Eq(zeta, 1)
    sol2 = solve(eqn2, K)
    print(sol2)
    eqn3 = Eq(zeta, 99999)
    sol3 = solve(eqn3, K)
    print(sol3)
    return

def q1_d():
    t = np.arange(0, 15, 0.0001)
    K = [0.5, 8, 300]
    for k in K:
        if k > 19:
            num = [k]
            den = [4, 5, (1+k)]
        else:
            num = [k / (1 + k)]
            den = [4 / (1 + k), 5 / (1 + k), 1]
        g = control.tf(num, den)
        resp, time = control.step(g, t)
        plt.plot(time, resp * 5, label=f'K = {k}') # multiply response by five to account for step = 5
    plt.title("System responses to three K values")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("response")
    plt.show()
    return

def q2_a():
    k = symbols('k')
    a = 67
    zeta = (25+a) / (2*sqrt(25*a + 100 * k))
    print(zeta)
    overshoot = -pi * zeta / sqrt(1-zeta**2)
    eqn = Eq(overshoot, ln(0.10))
    sol1 = solve(eqn, k)
    print(sol1)
    t = np.arange(0, 0.4, 0.0001)
    num = [4380]
    den = [1, 92, 6055]
    g = control.tf(num, den)
    resp, time = control.step(g, t)
    plt.plot(time, resp)
    plt.title("system response to step function input")
    plt.xlabel("time (s)")
    plt.ylabel("response")
    plt.show()
    return


def main():
#    q1_a_to_c()
#    q1_d()
    q2_a()


if __name__ == "__main__":
    fire.Fire(main())

