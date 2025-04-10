import pandas as pd
import matplotlib.pyplot as plt

def load_and_process_data(cpu_file, gpu_file):
    """
    Load and process data from CSV files.

    Parameters:
    - cpu_file (str): Path to the CPU data CSV file.
    - gpu_file (str): Path to the GPU data CSV file.

    Returns:
    - cpu_summary (DataFrame): Processed CPU data summary.
    - gpu_summaries (dict): Dictionary of processed GPU data summaries by batch size.
    """
    # Load data
    cpu_data = pd.read_csv(cpu_file)
    gpu_data = pd.read_csv(gpu_file)

    # Process CPU data
    cpu_summary = cpu_data.groupby("size").mean()
    cpu_summary["total_time"] = cpu_summary["loading_time"] + cpu_summary["computation_time"]
    cpu_summary["throughput"] = cpu_summary["batch_size"] / cpu_summary["total_time"]

    # Process GPU data by batch size
    gpu_summaries = {}
    for batch_size in gpu_data["batch_size"].unique():
        summary = gpu_data[gpu_data["batch_size"] == batch_size].groupby("size").mean()
        summary["total_time"] = summary["loading_time"] + summary["computation_time"]
        summary["throughput"] = summary["batch_size"] / summary["total_time"]
        gpu_summaries[batch_size] = summary

    return cpu_summary, gpu_summaries

def plot_stacked_bar(summary, title, xlabel, ylabel):
    """
    Plot a stacked bar chart for loading and computation times.

    Parameters:
    - summary (DataFrame): Data to plot (must include loading_time and computation_time).
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    x = range(len(summary))
    labels = summary.index
    plt.figure()
    plt.bar(x, summary["loading_time"], label="Loading Time")
    plt.bar(x, summary["computation_time"], bottom=summary["loading_time"], label="Computation + Communication Time")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis="y")
    plt.show()

def plot_throughput(summary, title, xlabel, ylabel):
    """
    Plot a bar chart for throughput.

    Parameters:
    - summary (DataFrame): Data to plot (must include throughput).
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    x = range(len(summary))
    labels = summary.index
    plt.figure()
    plt.bar(x, summary["throughput"], label="Throughput")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis="y")
    plt.show()

def plot_optimal_gpus(gpu_summaries):
    """
    Plot the optimal number of GPUs vs batch sizes.

    Parameters:
    - gpu_summaries (dict): Dictionary of GPU summaries by batch size.
    """
    optimal_gpus = []
    batch_sizes = sorted(gpu_summaries.keys())

    for batch_size in batch_sizes:
        summary = gpu_summaries[batch_size]
        optimal_gpus.append(summary["throughput"].idxmax())

    plt.figure()
    plt.plot(batch_sizes, optimal_gpus, marker="o", linestyle="--", label="Optimal GPUs")
    plt.xlabel("Batch Sizes")
    plt.ylabel("Optimal Number of GPUs")
    plt.title("Optimal Number of GPUs vs Batch Sizes")
    plt.yticks(range(0, max(optimal_gpus) + 1))  # Set y-axis ticks to integers only
    plt.grid()
    plt.legend()
    plt.show()

# Main execution
cpu_file = "results_cpu.csv"
gpu_file = "results_gpu.csv"

# Load and process data
cpu_summary, gpu_summaries = load_and_process_data(cpu_file, gpu_file)

# Generate plots
plot_stacked_bar(cpu_summary, "CPU: Time vs Number of Devices", "Number of Devices (CPU)", "Time (s)")
plot_throughput(cpu_summary, "CPU: Throughput vs Number of Devices", "Number of Devices (CPU)", "Throughput (images/s)")

for batch_size, gpu_summary in gpu_summaries.items():
    if batch_size == 128:
        # Include missing value for 1 GPU
        gpu_summary = gpu_summary.reindex(range(1, 3), fill_value=0)
    plot_stacked_bar(gpu_summary, f"GPU: Time vs Number of Devices (Batch Size = {batch_size})", "Number of Devices (GPU)", "Time (s)")
    plot_throughput(gpu_summary, f"GPU: Throughput vs Number of Devices (Batch Size = {batch_size})", "Number of Devices (GPU)", "Throughput (images/s)")

plot_optimal_gpus(gpu_summaries)