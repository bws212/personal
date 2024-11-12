import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    data = {
        'Task': ['Task A', 'Task B', 'Task C', 'Task D'],
        'Start': ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01'],
        'Finish': ['2021-01-31', '2021-02-28', '2021-03-31', '2021-04-30']
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Convert dates to pandas datetime format
    df['Start'] = pd.to_datetime(df['Start'])
    df['Finish'] = pd.to_datetime(df['Finish'])

    # Calculate the middle point of each task for the marker
    df['Middle'] = df['Start'] + (df['Finish'] - df['Start']) / 2

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the horizontal base line
    base_line_y = 0.5  # Y-coordinate of the base line
    ax.hlines(base_line_y, df['Start'].min(), df['Finish'].max(), color='grey', linewidth=2)

    # Plot each task as a vertical line from the base line with a marker
    for idx, row in df.iterrows():
        ax.plot([row['Start'], row['Finish']], [base_line_y, base_line_y], color='grey', linewidth=2)
        ax.plot(row['Middle'], base_line_y, 'o', markersize=8)
        ax.text(row['Middle'], base_line_y + 0.02, row['Task'], ha='center', va='bottom')

    # Format the x-axis to show dates nicely
    fig.autofmt_xdate()

    # Set the x-axis limits
    ax.set_xlim(df['Start'].min() - pd.Timedelta(days=10), df['Finish'].max() + pd.Timedelta(days=10))

    # Remove y-axis labels and ticks
    ax.yaxis.set_visible(False)

    # Set title
    plt.title('Project Timeline')

    # Show the plot
    plt.show()
    return

if __name__ == "__main__":
    main()