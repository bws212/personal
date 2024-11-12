import numpy as np
import fire
import matplotlib.pyplot as plt

def plotting(T, GT, RT):
    plt.plot(T, GT, label='GT', color='blue')
    plt.plot(T, RT, label='RT', color='red')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Heat Change (K * g / sec)')
    plt.xlim(550, 650)
    plt.legend()
    plt.show()
    return


def main():
    T = np.arange(423, 800, 0.5)
    T0 = 423.2 #K
    W = 100  #kg
    R1 = 8.314 #J/mol*K
    R2 = 0.0821 #L*atm/mole*K
    P = 1 #atm
    ynap = 0.01
    yair = 0.99
    Fa0 = 0.12 #mol*g/sec
    dhr1 = -1881 * 10**3
    dhr2 = -3282 * 10**3
    Cp = 1040 #J/kg*K
    E1 = 3.5 * 10**5
    E2 = 1.65 * 10**5
    MWair = 28.9 #g/mol
    MWNap = 128 #g/mol
    Fsum = (Fa0 * MWNap * 1 + Fa0 * MWair * 99) * 10**-3 #kg
    k1 = (1.61 * 10**33) * np.exp(-E1 / (R1*T))
    k2 = (5.14 * 10**13) * np.exp(-E2 / (R1 * T))
    CNap0 = (P * ynap) / (R2 * T)
    v = Fa0 / CNap0
    tao = W / v
    Ca = (CNap0 / (k1 * tao + 1))
    Fa = v * Ca
    Fb = v * ((k1 * tao * CNap0) / ((1 + k1 * tao) * (1 + k2 * tao)))
    GT = -dhr1 * (Fa0 - Fa) + (-dhr2) * (Fa0 - Fa - Fb)
    RT = (Fsum * Cp) * (T - T0)
    plotting(T, GT, RT)
    T_index = np.where(T == 600)[0][0]
    xa = (CNap0[T_index] - Ca[T_index]) / CNap0[T_index]
    print(xa)
    print(Fb[T_index])
    print(Fa0 - Fa[T_index] )
    S = Fb[T_index] / (Fa0 - Fa[T_index])
    print(S)
    Tc = 460
    UAh = (GT[T_index] - RT[T_index]) / (600 - Tc)
    RT2 = RT + (UAh * (T - Tc))
    plotting(T, GT, RT2)
    print(UAh)
    return


if __name__ == "__main__":
    fire.Fire(main())
