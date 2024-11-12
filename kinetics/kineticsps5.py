import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint

def plot(x, y):
    plot = plt.plot(x,y)
    plt.show()
    return plot


def k_value(space, frac_conv):
    k_values = []
    kt = []
    for i in [0, 1, 2, 3]:
        k = -math.log(1 - frac_conv[i]) / space[i]
        ktime = -math.log(1 - frac_conv[i])
        kt.append(ktime)
        k_values.append(k)
    return k_values, kt


def model(Cr, t, k1, k2, Ca0):
    dCrdt = k1*Ca0*math.exp(-k1*t) - k2*Cr
    return dCrdt




def q4_conc(r_volume, vol_flowA, vol_flow_solv, ca_initial, k_4, cr_initial):
    vol_flow_sum = vol_flow_solv + vol_flowA
    tao = r_volume/vol_flow_sum
    xa = ((tao)* k_4) / (1 + tao * k_4)
    print(f"xa is {xa}")
    ca_4 = ca_initial * (1 - xa)
    print(f"value of ca is {ca_4}")
    cr_4 = (cr_initial / ((tao)*k_4+1)) + ((tao) * k_4 * ca_4) / ((tao)*k_4 + 1)
    print(f"cr is {cr_4}")
    cs_4 = 0.25 - ca_4 - cr_4
    print(f"cs is {cs_4}\n")
    return ca_4, cr_4, cs_4


if __name__ == "__main__":
    #Question 1
    space_time = np.array([1, 5, 20, 30])
    fractional_conv = np.array([0.32, 0.55, 0.90, 0.96])
    ks, kt = k_value(space_time, fractional_conv)
    plot((space_time * ks), kt)
    print(kt)


    #Question 2
    time = np.array([0, 500, 1000, 1500])
    H2_gen = np.array([0.00, 0.31, 0.62, 0.93])
    mass_NaBH4 = 6 #g
    mass_NaOH = 3 #g
    mass_H20 = 21 #g
    mass_Ru = 0.25 #g * wt%
    h2_1500_L = 0.93 / 22.4
    mols_NaBH4 = mass_NaBH4 / 37.83
    mols_NaOH = mass_NaOH / 40
    mols_H20 = mass_H20 / 18
    print(mols_NaBH4, mols_NaOH, mols_H20, h2_1500_L)
    NaBH4_cons_1500 = h2_1500_L / 4
    print(NaBH4_cons_1500)
    frac_conv_NaBH4 = NaBH4_cons_1500 / mols_NaBH4
    print(f"fractional conversion NaBH4, t=1500: {frac_conv_NaBH4}")
    k_Q2 = NaBH4_cons_1500 / (time[3] * mass_Ru)
    print(f"rate constant at 25 Celsius: {k_Q2}")
    # 4
    h2_out = 1 / 22.4
    nabh4_out = h2_out / 4
    Fa0xa = nabh4_out
    new_k = k_Q2 * 60
    print(f"h2 out is {h2_out}, Fa0xa is {Fa0xa}, new k value is {new_k}")
    cat_weight = Fa0xa / new_k
    print(f"cat_weight is {cat_weight}")

    #Question 3
    ca0 = 1
    cr0 = 0.20
    cs0 = 0
    k1 = 0.025
    k2 = 0.010
    tao = 100
    ca = ca0 * math.exp(-k1 * tao)
    print(f"ca is {ca}")
    t = np.linspace(0,100,1000)
    cr = odeint(model, cr0, t, args=(k1, k2, ca0))
    print(f"Cr at time = 100 is {cr[-1][0]}")
    cs = ca0 + cr0 - ca - cr[-1][0]
    print(f"Cs at time = 100 is {cs}")
    #if cr0 = 0
    new_cr0 = 0
    new_cr = odeint(model, new_cr0, t, args=(k1, k2, ca0))
    print(f"New cr at time = 100 is {new_cr[-1][0]}")
    new_cs = ca0 + new_cr0 - new_cr[-1][0] - ca
    print(f"New cs at time = 100 is {new_cs}")
    #to find max Cr
    max = np.argmax(cr)
    time_max = t[max]
    cr_at_max = cr[max][0]
    print(f"Cr at max time is {cr_at_max}")
    print(f"max time is {time_max}\n")

    #Question 4
    vol_flowA0 = 100
    volume1 = 150
    volume2 = 1000
    ca0_4 = 0.25
    vol_flow_solv = 200
    T = 200
    k_4 = 0.25
    cr_init = 0
    #part 1
    print("concentrations for question 4, part 1, with small than large reactor")
    ca_4, cr_4, cs_4 = q4_conc(volume1, vol_flowA0, vol_flow_solv, ca0_4, k_4, cr_init)
    ca2_4, cr2_4, cs2_4 = q4_conc(volume2, vol_flowA0, vol_flow_solv, ca_4, k_4, cr_4)

    print("switched reactor order:")
    ca_4_switch, cr_4_switch, cs_4_switch = q4_conc(volume2, vol_flowA0, vol_flow_solv, ca0_4, k_4, cr_init)
    ca2_4_switch, cr2_4_switch, cs2_4_switch = q4_conc(volume1, vol_flowA0, vol_flow_solv, ca_4_switch,
                                                       k_4, cr_4_switch)
    print(f"the concentrations for either reactor order is the same: {cr2_4 == cr2_4_switch}")

    t = np.linspace(0,100,1000)
    cr_q4 = odeint(model, cr_4, t, args=(k_4, k_4, ca_4))
    max_4 = np.argmax(cr_q4)
    time_max_4 = t[max_4]
    cr_at_max_4 = cr_q4[max_4][0]
    print(f"Cr at max time is {cr_at_max_4}")
    print(f"max time is {time_max_4}\n")
    necessary_volume = time_max_4 * (vol_flowA0 + vol_flow_solv)
    print(f"necessary volume is {necessary_volume}\n")
    print(f"boost in production of Cr is {cr_at_max_4/cr2_4}")
