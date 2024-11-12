import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


T = 298 #K
P = 1 #atm
rp = 100e-9 #m
m = (108.1 * 6.022e-23) * 10e-3 #kg
kb = 1.38064852e-23 #J/K
Dg = 8.5 * 10**-6 #m^2 /s
rs = rp * math.exp(2.5 * 0.3**2) #m
delta_t = 8.5 * 60 #sec
omega = math.sqrt((8 * kb * T) / (math.pi * m)) #m/s
Kn = ((3 * Dg) / (omega * rs))

data = [
    [6.48E-07, 18.2, 20.1],
    [8.14E-07, 17.8, 20.2],
    [5.10E-07, 19.7, 19.7],
    [7.04E-07, 19.2, 21.4],
    [7.45E-07, 19.5, 19.6],
    [7.10E-07, 16.1, 17.9],
    [6.82E-07, 19.4, 21.5],
    [6.52E-07, 20.8, 20.9],
    [6.91E-07, 21.9, 22],
    [5.54E-07, 17.9, 19.5],
    [9.80E-07, 20.2, 20.4],
    [9.72E-07, 17.9, 17.9],
    [9.80E-07, 19.0, 22],
    [7.01E-07, 20.1, 20.1]
]
columns = ['Sa (cm2 cm-3)', 'N2O5 [ppb] Measured @outlet of flow tube, (WITH PARTICLES)', 'N2O5 [ppb] Measured @outlet of flow tube, baseline w/filter bypass (NO PARTICLES)']

# Create DataFrame
df = pd.DataFrame(data, columns=columns)


def plotting_khet(df):
    k_het = -(1 / delta_t) * np.log(df['N2O5 [ppb] Measured @outlet of flow tube, (WITH PARTICLES)'] /
                                    df['N2O5 [ppb] Measured @outlet of flow tube, baseline w/filter bypass (NO PARTICLES)'])

    above_threshold = k_het[k_het > 0.0001]
    below_threshold = k_het[k_het <= 0.0001]

    plt.figure(figsize=(8, 6))
    plt.plot(df['Sa (cm2 cm-3)'], k_het, 'ko')
    plt.plot(df['Sa (cm2 cm-3)'][k_het > 0.0001], above_threshold, 'ro', markersize=10)
    plt.plot(df['Sa (cm2 cm-3)'][k_het <= 0.0001], below_threshold, 'bo', markersize = 10)

    for i in range(len(df)):
        if k_het[i] > 0.0001:
            plt.plot([0, df['Sa (cm2 cm-3)'][i]], [0, k_het[i]], 'r--')
        else:
            plt.plot([0, df['Sa (cm2 cm-3)'][i]], [0, k_het[i]], 'b--')

    plt.xlabel('Sa (cm2 cm-3)', fontsize=14)
    plt.ylabel('k_het (s-1)', fontsize=14)
    plt.title('k_het vs. Sa', fontsize=16)
    plt.grid(True)
    plt.show()

    df['gamma'] = (4 * k_het) / (omega * df['Sa (cm2 cm-3)'])
    print(df['gamma'])
    return


if __name__ == '__main__':
    print(omega)
    print(Kn)
    plotting_khet(df)
    print(df.to_string(index=False))





