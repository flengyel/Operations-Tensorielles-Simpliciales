# main_experiment.py

import matplotlib
matplotlib.use('TkAgg')  # Set the backend before importing pyplot

import numpy as np
from scipy import stats
import csv
from typing import Tuple, List
from tensor_ops import random_real_tensor, bdry, degen, ___SEED___  # Ensure tensor_ops.py is in the same directory
from tqdm import tqdm  # For progress bars
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import ast  # For safe evaluation of strings to lists

# ----------------------------
# Logging Configuration
# ----------------------------

# Define the log file path
log_file_path = os.path.join(os.getcwd(), "experiment.log")

# Reset any existing logging handlers to avoid duplication
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging to output to both console and a log file
logging.basicConfig(
    level=logging.INFO,  # Capture INFO and above messages
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ----------------------------
# Tensor Reconstruction and Loss Computation
# ----------------------------

def reconstruct_T_from_S(S: np.ndarray, T_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reconstruct an approximation of T from S by degenerating (duplicating) elements along each axis.

    Parameters:
        S (np.ndarray): Tensor of shape (a_1 - 1, a_2 - 1, ..., a_k - 1)
        T_shape (tuple): Shape of T (a_1, a_2, ..., a_k)

    Returns:
        np.ndarray: Reconstructed tensor of shape T_shape
    """
    # Apply degen with k=0 to duplicate the 0-th hypercolumn along each axis
    T_hat = degen(S, 0)
    
    # After degeneracy, ensure that T_hat matches T_shape
    if T_hat.shape != T_shape:
        raise ValueError(f"Reconstructed tensor shape {T_hat.shape} does not match target shape {T_shape}.")
    
    return T_hat

def compute_loss(T: np.ndarray, S: np.ndarray) -> float:
    """
    Compute the loss between T and the reconstructed T from S.

    Parameters:
        T (np.ndarray): Original tensor of shape T_shape
        S (np.ndarray): Tensor of shape (a_1 - 1, a_2 - 1, ..., a_k - 1)

    Returns:
        float: The norm of the difference between T and reconstructed T_hat
    """
    T_hat = reconstruct_T_from_S(S, T.shape)
    T_norm = T / np.linalg.norm(T) if np.linalg.norm(T) != 0 else T
    T_hat_norm = T_hat / np.linalg.norm(T_hat) if np.linalg.norm(T_hat) != 0 else T_hat
    loss = np.linalg.norm(T_norm - T_hat_norm)
    return loss

# ----------------------------
# Experiment Execution
# ----------------------------

def run_experiment(
    tensor_shape: Tuple[int, ...],
    num_trials: int = 10000,
    mean: float = 0.0,
    std: float = 1.0
) -> dict:
    """
    Run the experiment for a specific tensor shape.

    Parameters:
        tensor_shape (Tuple[int, ...]): Shape of tensor T.
        num_trials (int): Number of trials to run.
        mean (float): Mean of the normal distribution for tensor generation.
        std (float): Standard deviation of the normal distribution for tensor generation.

    Returns:
        dict: A dictionary containing experiment results.
    """
    logging.info(f"Starting experiments for tensor shape: {tensor_shape}")
    losses_S = []
    losses_boundary = []
    for trial in tqdm(range(num_trials), desc=f"Tensor Shape {tensor_shape}", unit="trial"):
        # Unique seed for T to ensure reproducibility
        seed_T = trial + 12345
        T = random_real_tensor(shape=tensor_shape, mean=mean, std=std, seed=seed_T)
        
        # Compute the boundary tensor ∂T
        partial_T = bdry(T)
        S_shape = partial_T.shape
        
        # Generate S using the global RNG without specifying a seed to ensure independence
        S = random_real_tensor(shape=S_shape, mean=mean, std=std, seed=___SEED___)
        
        # Compute losses
        loss_S = compute_loss(T, S)
        loss_boundary = compute_loss(T, partial_T)
        
        losses_S.append(loss_S)
        losses_boundary.append(loss_boundary)

    # Convert to numpy arrays for efficient computation
    losses_S = np.array(losses_S)
    losses_boundary = np.array(losses_boundary)

    # Calculate statistical metrics
    avg_loss_S = np.mean(losses_S)
    avg_loss_boundary = np.mean(losses_boundary)
    inequality_holds = np.sum(losses_S >= losses_boundary)
    inequality_percentage = (inequality_holds / num_trials) * 100
    std_loss_S = np.std(losses_S, ddof=1)
    std_loss_boundary = np.std(losses_boundary, ddof=1)
    t_statistic, p_value = stats.ttest_rel(losses_S, losses_boundary)
    mean_diff = avg_loss_S - avg_loss_boundary
    pooled_std = np.sqrt((std_loss_S ** 2 + std_loss_boundary ** 2) / 2)
    cohen_d = mean_diff / pooled_std if pooled_std != 0 else np.inf

    # Determine per-axis parity
    axis_parities = ['Even' if dim % 2 == 0 else 'Odd' for dim in tensor_shape]
    num_even_axes = axis_parities.count('Even')
    num_odd_axes = axis_parities.count('Odd')

    # Compile results into a dictionary
    results_dict = {
        'Tensor Shape': tensor_shape,
        'Number of Axes': len(tensor_shape),
        'Axis Parities': axis_parities,
        'Number of Even Axes': num_even_axes,
        'Number of Odd Axes': num_odd_axes,
        'Average Loss T & S': avg_loss_S,
        'Average Loss T & ∂T': avg_loss_boundary,
        'Inequality Holds (%)': inequality_percentage,
        'Standard Deviation Loss S': std_loss_S,
        'Standard Deviation Loss ∂T': std_loss_boundary,
        'Paired t-test Statistic': t_statistic,
        'p-value': p_value,
        'Cohen\'s d': cohen_d
    }

    logging.info(f"Completed experiments for tensor shape: {tensor_shape}")
    logging.info(f"  Number of Even Axes: {num_even_axes}")
    logging.info(f"  Number of Odd Axes: {num_odd_axes}")
    logging.info(f"  Average Loss between T and S: {avg_loss_S:.6f}")
    logging.info(f"  Average Loss between T and ∂T: {avg_loss_boundary:.6f}")
    logging.info(f"  Inequality Loss(T, S) ≥ Loss(T, ∂T) holds in {inequality_percentage:.2f}% of trials")
    logging.info(f"  Standard Deviation of Loss(T, S): {std_loss_S:.6f}")
    logging.info(f"  Standard Deviation of Loss(∂T): {std_loss_boundary:.6f}")
    logging.info(f"  Paired t-test Statistic: {t_statistic:.6f}")
    logging.info(f"  p-value: {p_value}")
    logging.info(f"  Cohen's d (Effect Size): {cohen_d:.6f}")
    logging.info("-" * 60)

    return results_dict

# ----------------------------
# Plotting Functionality
# ----------------------------

def plot_results(csv_file: str):
    """
    Generate and save plots based on the experiment results stored in a CSV file.

    Parameters:
        csv_file (str): Path to the CSV file containing experiment results.
    """
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        logging.error(f"CSV file '{csv_file}' not found. Skipping plotting.")
        return

    # Load the results
    df = pd.read_csv(csv_file)

    # Ensure that 'Tensor Shape' is treated as a categorical variable
    df['Tensor Shape'] = df['Tensor Shape'].astype(str)

    # Parse axis parities from the 'Axis Parities' column safely
    df['Axis Parity List'] = df['Axis Parities'].apply(lambda x: ast.literal_eval(x))

    # Calculate the number of even and odd axes
    df['Number of Even Axes'] = df['Axis Parity List'].apply(lambda x: x.count('Even'))
    df['Number of Odd Axes'] = df['Axis Parity List'].apply(lambda x: x.count('Odd'))

    # Set plot style
    sns.set(style="whitegrid")

    # Create 'plots' directory if it doesn't exist
    plots_dir = os.path.join(os.getcwd(), "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        logging.info(f"Created 'plots' directory at {plots_dir}")

    # ----------------------------
    # Plot 1: Average Loss Comparison
    # ----------------------------
    plt.figure(figsize=(14, 7))
    df_melted = df.melt(id_vars=["Tensor Shape"], value_vars=["Average Loss T & S", "Average Loss T & ∂T"],
                        var_name="Condition", value_name="Average Loss")
    sns.barplot(x="Tensor Shape", y="Average Loss", hue="Condition", data=df_melted, palette=["skyblue", "lightgreen"])
    plt.title("Average Loss Comparison Across Tensor Shapes")
    plt.xlabel("Tensor Shape")
    plt.ylabel("Average Loss")
    plt.xticks(rotation=45)
    plt.legend(title="Condition")
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "average_loss_comparison.png")
    plt.savefig(plot_path)
    plt.show(block=True)  # Display the plot window
    plt.close()
    logging.info(f"Plot 'average_loss_comparison.png' has been saved to {plots_dir} and displayed on screen.")

    # ----------------------------
    # Plot 2: Inequality Holding Percentage vs. Number of Even Axes
    # ----------------------------
    plt.figure(figsize=(14, 7))
    sns.scatterplot(x="Number of Even Axes", y="Inequality Holds (%)", hue="Number of Odd Axes",
                    palette="viridis", size="Number of Odd Axes", sizes=(50, 200), data=df, alpha=0.7)
    plt.title("Inequality Holds (%) vs. Number of Even Axes")
    plt.xlabel("Number of Even Axes")
    plt.ylabel("Inequality Holds (%)")
    plt.legend(title="Number of Odd Axes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "inequality_holds_vs_even_axes.png")
    plt.savefig(plot_path)
    plt.show(block=True)
    plt.close()
    logging.info(f"Plot 'inequality_holds_vs_even_axes.png' has been saved to {plots_dir} and displayed on screen.")

    # ----------------------------
    # Plot 3: Paired t-test Statistic and Cohen's d vs. Number of Even Axes
    # ----------------------------
    plt.figure(figsize=(14, 7))
    df_plot = df[['Tensor Shape', 'Paired t-test Statistic', 'Cohen\'s d', 'Number of Even Axes']]
    # Melt for easier plotting
    df_plot_melted = df_plot.melt(id_vars=['Tensor Shape', 'Number of Even Axes'], value_vars=['Paired t-test Statistic', 'Cohen\'s d'],
                                  var_name='Metric', value_name='Value')
    sns.barplot(x='Number of Even Axes', y='Value', hue='Metric', data=df_plot_melted, palette=["salmon", "lightblue"])
    plt.title("Paired t-test Statistic and Cohen's d vs. Number of Even Axes")
    plt.xlabel("Number of Even Axes")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=45)
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "statistical_metrics_vs_even_axes.png")
    plt.savefig(plot_path)
    plt.show(block=True)
    plt.close()
    logging.info(f"Plot 'statistical_metrics_vs_even_axes.png' has been saved to {plots_dir} and displayed on screen.")

    # ----------------------------
    # Plot 4: Cohen's d (Effect Size) vs. Number of Even Axes
    # ----------------------------
    plt.figure(figsize=(14, 7))
    sns.scatterplot(x="Number of Even Axes", y="Cohen's d", hue="Number of Odd Axes",
                    palette="Set2", size="Number of Odd Axes", sizes=(50, 200), data=df, alpha=0.7)
    plt.title("Cohen's d (Effect Size) vs. Number of Even Axes")
    plt.xlabel("Number of Even Axes")
    plt.ylabel("Cohen's d")
    plt.legend(title="Number of Odd Axes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "cohens_d_vs_even_axes.png")
    plt.savefig(plot_path)
    plt.show(block=True)
    plt.close()
    logging.info(f"Plot 'cohens_d_vs_even_axes.png' has been saved to {plots_dir} and displayed on screen.")

    # ----------------------------
    # Plot 5: Distribution of Axis Parity
    # ----------------------------
    plt.figure(figsize=(10, 6))
    # Create a single list of all axis parities
    all_parities = [parity for sublist in df['Axis Parity List'] for parity in sublist]
    sns.countplot(x=all_parities)  # Removed 'palette' to fix warnings
    plt.title("Distribution of Axis Parity Across All Tensor Axes")
    plt.xlabel("Axis Parity")
    plt.ylabel("Count")
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "axis_parity_distribution.png")
    plt.savefig(plot_path)
    plt.show(block=True)
    plt.close()
    logging.info(f"Plot 'axis_parity_distribution.png' has been saved to {plots_dir} and displayed on screen.")

    # ----------------------------
    # Plot 6: Average Loss vs. Number of Even Axes
    # ----------------------------
    plt.figure(figsize=(14, 7))
    sns.boxplot(x="Number of Even Axes", y="Average Loss T & S", data=df, showfliers=False, color='skyblue')
    sns.boxplot(x="Number of Even Axes", y="Average Loss T & ∂T", data=df, showfliers=False, color='lightgreen')
    plt.title("Average Loss vs. Number of Even Axes")
    plt.xlabel("Number of Even Axes")
    plt.ylabel("Average Loss")
    plt.legend(labels=["Average Loss T & S", "Average Loss T & ∂T"], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "average_loss_vs_even_axes.png")
    plt.savefig(plot_path)
    plt.show(block=True)
    plt.close()
    logging.info(f"Plot 'average_loss_vs_even_axes.png' has been saved to {plots_dir} and displayed on screen.")

    # ----------------------------
    # Plot 7: Heat Map of Effect Sizes (Cohen's d)
    # ----------------------------
    plt.figure(figsize=(10, 8))
    # Pivot the data to create a matrix suitable for heatmap
    # Rows: Number of Even Axes
    # Columns: Number of Odd Axes
    # Values: Average Cohen's d
    heatmap_data = df.groupby(['Number of Even Axes', 'Number of Odd Axes'])['Cohen\'s d'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Number of Even Axes', columns='Number of Odd Axes', values='Cohen\'s d')
    
    # Create the heatmap
    sns.heatmap(heatmap_pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': "Cohen's d"})
    plt.title("Heat Map of Effect Sizes (Cohen's d)")
    plt.xlabel("Number of Odd Axes")
    plt.ylabel("Number of Even Axes")
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "effect_size_heatmap.png")
    plt.savefig(plot_path)
    plt.show(block=True)
    plt.close()
    logging.info(f"Plot 'effect_size_heatmap.png' has been saved to {plots_dir} and displayed on screen.")

    logging.info("All plots have been generated, saved to the 'plots' directory, and displayed on screen.")

# ----------------------------
# Main Execution Function
# ----------------------------

def main():
    """
    Main function to execute the experiments, save results, and generate plots.
    """
    try:
        # Define the tensor shapes to test (both isotropic and anisotropic)
        tensor_shapes = [
            (5, 5, 5),       # 0 even axes
            (4, 5, 5),       # 1 even axis
            (5, 6, 7),       # 1 even axis
            (4, 6, 8),       # 3 even axes
            (5, 5, 5),       # 0 even axes
            (6, 7, 8, 9),    # 2 even axes
            (8, 9, 10),      # 2 even axes
            (5, 7, 9, 10),   # 1 even axis
            (5,7,9,11),
            (6, 6, 6, 6),      # 4 even axes
            (7,7,7,7),
            (9,9,9),
            (9,9,11,11)
        ]

        num_trials = 10000  # Number of trials per tensor shape
        mean = 0.0           # Mean of the normal distribution
        std = 1.0            # Standard deviation of the normal distribution

        all_results = []     # List to store results for all tensor shapes

        for shape in tensor_shapes:
            result = run_experiment(tensor_shape=shape, num_trials=num_trials, mean=mean, std=std)
            all_results.append(result)
        
        # Save all_results to a CSV file for further analysis
        csv_file = 'experiment_results.csv'
        csv_columns = ['Tensor Shape', 'Number of Axes', 'Axis Parities', 'Number of Even Axes', 'Number of Odd Axes',
                      'Average Loss T & S', 'Average Loss T & ∂T',
                      'Inequality Holds (%)', 'Standard Deviation Loss S',
                      'Standard Deviation Loss ∂T', 'Paired t-test Statistic',
                      'p-value', 'Cohen\'s d']

        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in all_results:
                    # Convert tuple to string for CSV compatibility
                    data_copy = data.copy()
                    data_copy['Tensor Shape'] = str(data_copy['Tensor Shape'])
                    writer.writerow(data_copy)
            logging.info(f"All results have been saved to {csv_file}")
        except IOError:
            logging.error("I/O error while trying to write the CSV file.")

        # Generate plots based on the saved CSV
        plot_results(csv_file)

    except Exception as e:
        logging.exception("An unexpected error occurred during the experiments.")
        print("An error occurred. Check experiment.log for details.")

if __name__ == "__main__":
    main()
