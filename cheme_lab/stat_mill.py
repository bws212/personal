import fire
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import glob
import os


def plot_feret_distribution(data):
    save_folder = os.path.join(os.getcwd(), "stat_mill_hist")
    experiment_title = None
    for dat in data:
        results_df = dat[1]
        if results_df is not None and "Feret" in results_df.columns:
            experiment_title = results_df["Title"][0] if "Title" in results_df.columns else "Unknown Experiment"
            feret_values = results_df["Feret"].to_numpy()

            bins = 60
            plt.figure(figsize=(8, 5))
            plt.hist(feret_values, bins=bins, alpha=0.7, color="blue", edgecolor="black")
            plt.xlabel("Feret Diameter (mm)")
            plt.ylabel("Frequency")
            plt.title(f"Feret Diameter Distribution - {experiment_title}")
            plt.grid(True)
            safe_title = experiment_title.replace(" ", "_").replace("/", "_")
            save_path = os.path.join(save_folder, f"{safe_title}_hist.png")
            plt.savefig(save_path)
            plt.close()
            plt.show()


            print(f"Saved histogram for {experiment_title} at {save_path}")
        else:
            print(f"Missing results file or 'Feret' column for {experiment_title}")
    return


def average_feret(data):
    for dat in data:
        summary_df = dat[0]
        if summary_df is not None and "Feret" in summary_df.columns:
            experiment_title = summary_df["Title"][0] if "Title" in summary_df.columns else "Unknown Experiment"

            avg_feret = summary_df["Feret"].mean()
            print(f"Average Feret Diameter for {experiment_title}: {avg_feret:.3f}")
        else:
            print("Missing summary file or 'Feret' column not found")
    return


def pair_experiment_files(directory):
    file_paths = sorted(glob.glob(os.path.join(directory, "*.csv")))
    experiment_pairs = {}
    for file in file_paths:
        filename = os.path.basename(file)

        if "_summ.csv" in filename:
            base_name = filename.replace("_summ.csv", "")
            key = f"{base_name}_summ"
        elif "_results.csv" in filename:
            base_name = filename.replace("_results.csv", "")
            key = f"{base_name}_res"
        else:
            continue

        df = pl.read_csv(file).with_columns(pl.lit(key).alias("Title"))

        if base_name not in experiment_pairs:
            experiment_pairs[base_name] = [None, None]  # [summary, results]

        if "_summ.csv" in filename:
            experiment_pairs[base_name][0] = df  # Summary file
        elif "_results.csv" in filename:
            experiment_pairs[base_name][1] = df  # Results file
    paired_data = list(experiment_pairs.values())

    return paired_data


def histograms(bins, freq):
    bar_width = 0.8
    plt.figure(figsize=(8, 5))
    plt.bar(bins, freq, width=bar_width, color="orange", edgecolor="black", alpha=0.7)
    plt.xlabel("Characteristic Length (mm)")
    plt.ylabel("Normalized Frequency")
    plt.title("Red Bean Normalized Frequency by Characteristic Length")
    plt.grid(axis='y', alpha=0.7)
    plt.show()
    return


def norm_weight_hist():
    #Red Bean Data
    # weight = np.array([0.524, 0.629, 0.678, 0.639, 0.666, 0.57, 0.613, 0.538, 0.534, 0.353,
    #                    0.525, 0.718, 0.487, 0.477, 0.525, 0.638, 0.427, 0.484, 0.491, 0.468])
    #
    # characteristic_length = np.array([13.1, 12.98, 13.555, 12.81, 13.845, 13.15, 12.83, 12.44,
    #                                   12.5, 10.44, 12.085, 13.46, 12.315, 11.41, 12.3, 14.2,
    #                                   11.745, 11.965, 12.58, 11.875])
    # White Corn Data
    weight = np.array([0.789, 1.032, 0.608, 0.967, 1.22, 0.914, 1.271, 0.822, 1.177, 1.012,
                       1.094, 0.925, 1.398, 0.88, 1.045, 0.922, 0.816, 0.87, 0.722, 0.978])
    characteristic_length = np.array([15.905, 19.125, 16.845, 17.745, 19.34, 18.015, 19.33, 16.9, 20.105, 20.63,
                                      16.69, 15.625, 19.555, 15.64, 17.595, 17.43, 13.765, 17.755, 14.935, 16.86])
    bins = np.histogram_bin_edges(characteristic_length, bins="auto")

    bin_indices = np.digitize(characteristic_length, bins) - 1
    binned_weights = [weight[bin_indices == i].mean() if np.any(bin_indices == i) else 0 for i in range(len(bins) - 1)]
    binned_weights = np.array(binned_weights) / np.sum(binned_weights)

    plt.figure(figsize=(8, 5))
    plt.bar(bins[:-1], binned_weights, width=np.diff(bins), align="edge", color="orange", edgecolor="black", alpha=0.7)
    plt.xlabel("Characteristic Length (mm)")
    plt.ylabel("Normalized Weight (Probability Density)")
    plt.title("White Corn Normalized Histogram of Weight vs. Characteristic Length ")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.show()
    return


def main():
    directory = "stat_mill_data/"  # Adjust path
    data = pair_experiment_files(directory)
    for i, (summ, results) in enumerate(data, 1):
        summ_title = summ["Title"][0] if summ is not None else "Missing"
        results_title = results["Title"][0] if results is not None else "Missing"
        print(f"Experiment {i}: Summary File -> {summ_title}, Results File -> {results_title}")
    average_feret(data)
    plot_feret_distribution(data)

    WC_bins = [11, 12, 13, 14, 15]
    WC_normalized_frequency = [0.05, 0.2, 0.45, 0.25, 0.05]
    histograms(WC_bins, WC_normalized_frequency)
    norm_weight_hist()
    return


if __name__ == "__main__":
    fire.Fire(main())