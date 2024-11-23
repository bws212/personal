import numpy as np
import matplotlib.pyplot as plt
import polars as pl


### Global Variables
t = 2  # hours
extract_factor = 0.98  # REE extraction efficiency (70%)
kg_ton = 907.185  # kg per ton
CFA_prod = 30 * 10 ** 6  # tons of CFA produced per year
CFA_prod_kg = CFA_prod * kg_ton  # Convert to kg
market_share = 0.10  # 10% of the market
ree_conc_in_cfa = 0.0234 / 100  # REE concentration in CFA (% converted to fraction)
price_Nd = 5000  # $/kg for Neodymium
solvent_cost_per_L = 10 # $ per liter
solvent_recycling_rate = 0.995  # 99% recycling


def scaler(scale, CFA, solv, scf):
    CFA_scaled = CFA * scale
    solv_scaled = solv * scale
    scf_scaled = scf * scale
    size = scf_scaled  # Assuming reactor size is proportional to scf volume
    return CFA_scaled, solv_scaled, scf_scaled, size


def reactor_balance(CFA, solv, scf):
    ree_in_reactor = CFA * ree_conc_in_cfa
    ree_out = ree_in_reactor * extract_factor  # Amount of REE extracted
    revenue = ree_out * price_Nd * 12 * 300  # Revenue generated from extracted REE
    new_solvent_needed = solv * (1 - solvent_recycling_rate)
    operating_cost = new_solvent_needed * solvent_cost_per_L * 12 * 300 # Operating cost for solvent
    return ree_out, revenue, operating_cost


def main():
    m_ree_tons = 350000  # Total REE market in tons/year
    m_ree = m_ree_tons * kg_ton  # Convert to kg/year
    mass = m_ree * market_share  # Mass needed for 10% market share
    print(f"Mass of REE needed for 10% of market share: {mass:.2e} kg/year\n")
    alpha = 3.25  # Selectivity (not used in current calculations)
    solv_initial = 20e-3  # L (initial solvent volume)
    CFA_initial = 2e-3  # kg (initial CFA mass)
    scf_initial = 250e-3  # L (initial supercritical fluid volume)
    scale_factors = np.array([10, 100, 500, 1000, 5000, 10000])
    scale_factors_list = []
    reactor_sizes_list = []
    revenues_list = []
    operating_costs_list = []
    profits_list = []

    for scale in scale_factors:
        CFA_scaled, solv_scaled, scf_scaled, size = scaler(scale, CFA_initial, solv_initial, scf_initial)
        ree_out, revenue, operating_cost = reactor_balance(CFA_scaled, solv_scaled, scf_scaled)
        profit = revenue - operating_cost  # Calculate profit

        scale_factors_list.append(scale)
        reactor_sizes_list.append(size)
        revenues_list.append(revenue)
        operating_costs_list.append(operating_cost)
        profits_list.append(profit)

    data = {
        'Scale Factor': scale_factors_list,
        'Reactor Size (L)': reactor_sizes_list,
        'Revenue ($)': revenues_list,
        'Operating Cost ($)': operating_costs_list,
        'Profit ($)': profits_list
    }
    df = pl.DataFrame(data)
    print(df)

    plt.figure(figsize=(8, 6))
    plt.plot(scale_factors, revenues_list, label='Revenue')
    plt.plot(scale_factors, operating_costs_list, label='Operating Cost')
    plt.xlabel('Scale Factor')
    plt.ylabel('Amount ($)')
    plt.title('Revenue and Operating Cost vs. Scale Factor')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()