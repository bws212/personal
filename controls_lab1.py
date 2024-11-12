import pandas as pd
import fire
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sympy.printing.pretty.pretty_symbology import line_width


def print_params(title, params):
    print(f"{title}:")
    for controller_type, settings in params.items():
        print(f"{controller_type} Controller:")
        for param, value in settings.items():
            print(f"  {param}: {value:.3f}")
    print()


def cohen_coon_tuning(K, tau, theta):
    Kc_pi = 0.45 / K * (tau + 0.092 * theta) / (tau + 2.22 * theta)
    Ti_pi = 3.33 * theta * ((tau + 0.092 * theta) / (tau + 2.22 * theta))
    Kc_pid = (0.67 / K) * ((tau + 0.185 * theta) / (tau + 0.611 * theta))
    Ti_pid = 2.5 * theta * (tau + 0.185 * theta) / (tau + 0.611 * theta)
    Td_pid = 0.37 * theta * (tau / (tau + 0.185 * theta))
    return {
        "PI": {"Kc": Kc_pi, "Ti": Ti_pi},
        "PID": {"Kc": Kc_pid, "Ti": Ti_pid, "Td": Td_pid}
    }


def ziegler_nichols_tuning(K, tau, theta):
    Kc_p = 1 / (K * theta)
    Kc_pi = 0.9 / (K * theta)
    Ti_pi = theta / 0.3
    Kc_pid = 1.2 / (K * theta)
    Ti_pid = 2 * theta
    Td_pid = 0.5 * theta
    return {
        "P": {"Kc": Kc_p},
        "PI": {"Kc": Kc_pi, "Ti": Ti_pi},
        "PID": {"Kc": Kc_pid, "Ti": Ti_pid, "Td": Td_pid}
    }


def foptd(t, K, tau, tau_d, T_initial):
    tau_d = max(0, tau_d)
    tau = max(0, tau)
    response = np.array([T_initial + K * (1 - np.exp(-(t - tau_d) / tau)) if t >= tau_d else T_initial for t in t])
    return response


def convert_time_to_seconds(time_val):
    """Convert a datetime.time value to total seconds."""
    return time_val.hour * 3600 + time_val.minute * 60 + time_val.second

def clean_column_names(df):
    df.columns = df.columns.str.replace(r'\n', ' ', regex=True)
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)
    df.columns = df.columns.str.strip()
    return df


def data_retrieve(names):
    data_objects = []
    for n in names:
        df = pd.read_excel(n)
        data_objects.append(df)
    return data_objects

def exp_plot(data):
    plt.figure(figsize=(10, 6))
    data['Elapsed Time (s)'] = data['Elapsed Time'].apply(convert_time_to_seconds)
    time = data['Elapsed Time (s)']
    temperature = data['Temperature T1 [C]']
    plt.plot(time, temperature, color = "blue", linewidth = 2)
    plt.xlabel("Elapsed Time (s)", fontsize = 14)
    plt.ylabel("Temperature T1 [C]", fontsize =14)
    plt.title("Temperature (C) vs. Elapsed Time (s)", fontsize = 14)
    plt.show()
    return

def trial_plot(data, time, temperature, K, tau, theta, T_initial):
    plt.figure(figsize=(10, 6))
    plt.plot(time, temperature, label="Experimental Data", color = "blue", linewidth = 2)
    plt.plot(time, foptd(time, K, tau, theta, T_initial), label="FOPTD Fit", linestyle="--", color = "red",
             linewidth = 2)
    plt.xlabel("Elapsed Time (s), offset by 650 seconds from true start time", fontsize = 14)
    plt.ylabel("Temperature T1 [C]", fontsize = 14)
    plt.title("FOPTD Model Fit to Experimental Data", fontsize = 14)
    plt.legend()
    plt.grid()
    plt.show()
    plt.show()
    return


def main():
    #Pulling data and cleaning it
    excel_file_names = ["lab1exp1trial2.xlsx", "lab1exp1trial3.xlsx"]
    tri2, tri3 = data_retrieve(excel_file_names)
    tri3 = clean_column_names(tri3)
    exp_plot(tri3)
    #Converting to seconds
    tri3['Elapsed Time (s)'] = tri3['Elapsed Time'].apply(convert_time_to_seconds)
    # Choosing step response component of curve
    step_data = tri3[(tri3['Elapsed Time (s)'] >= 650) & (tri3['Elapsed Time (s)'] <= 977)]
    time = step_data['Elapsed Time (s)'].values
    # Resetting time = 0sec
    time = np.array(time)
    time = time - 650
    # Selecting same T as t range
    temperature = step_data["Temperature T1 [C]"].values
    # initial guesses for curve fitting
    T_initial = temperature[0]
    K_initial = (temperature[330] - T_initial) / 0.25  # Adjust based on known step size
    tau_initial = 10
    theta_initial = 10
    initial_guess = [K_initial, tau_initial, theta_initial, T_initial]
    bounds = ([0, 0, 0, T_initial - 10], [np.inf, np.inf, np.inf, T_initial + 10])
    # pull FOPTD params from curve fit
    params, _ = curve_fit(foptd, time, temperature, p0=initial_guess, bounds=bounds)
    K, tau, theta, T_initial = params
    print(
        f"Estimated Parameters: Process Gain (K) = {K:.3f}, Time Constant (τ) = {tau:.3f}, Time Delay (θ) = {theta:.3f}")

    # plot FOPTD model and experimental data
    trial_plot(step_data, time, temperature, K, tau, theta, T_initial)

    # Plug in derived parameters into tuning system
    zn_params = ziegler_nichols_tuning(K, tau, theta)
    cc_params = cohen_coon_tuning(K, tau, theta)

    print_params("Ziegler-Nichols Parameters", zn_params)
    print_params("Cohen-Coon Parameters", cc_params)


if __name__ == "__main__":
    fire.Fire(main)