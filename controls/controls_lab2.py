import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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
        try:
            df = pd.read_csv(n)
            data_objects.append(df)
        except Exception as e:
            print(f"Error with file {n}: {e}")
    return data_objects


def exp_plot(data):
    plt.figure(figsize=(10, 6))
    data['Elapsed Time'] = pd.to_datetime(data['Elapsed Time'], format='%M:%S').dt.time
    data['Elapsed Time (s)'] = data['Elapsed Time'].apply(convert_time_to_seconds)

    time = data['Elapsed Time (s)']
    temperature = data['Flow F1 [l/min]']  # Adjusted based on your screenshot
    plt.plot(time, temperature, color="blue", linewidth=2)
    plt.xlabel("Elapsed Time (s)", fontsize=14)
    plt.ylabel("Flow Rate [L/min]", fontsize=14)
    plt.title("Flow Rate vs. Elapsed Time (s)", fontsize=14)
    plt.show()
    return


def main():
    csv_file_names = ["lab2exp1.csv",
                      "lab2exp2a.csv",
                      "lab2exp2pionly.csv",
                      "lab2exp2ponly.csv",
                      "lab2exp3parta.csv",
                      "lab2exp3partb.csv"]

    exp1, exp2a, exp2pionly, exp2ponly, exp3parta, exp3partb = data_retrieve(csv_file_names)
    exp1 = clean_column_names(exp1)
    exp2a = clean_column_names(exp2a)
    exp2pionly = clean_column_names(exp2pionly)
    exp2ponly = clean_column_names(exp2ponly)
    exp3parta = clean_column_names(exp3parta)
    exp3partb = clean_column_names(exp3partb)
    exp_plot(exp1)
    exp_plot(exp2a)
    exp_plot(exp2pionly)
    exp_plot(exp2ponly)
    exp_plot(exp3parta)
    exp_plot(exp3partb)

    return

if __name__ == '__main__':
    main()