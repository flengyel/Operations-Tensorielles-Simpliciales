# main_experiment.py

import numpy as np
from scipy import stats
import csv
from typing import Tuple
from tqdm import tqdm
from tensor_ops import random_real_tensor, bdry, ___SEED___  # Ensure tensor_ops.py is in the same directory
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def reconstruct_T_from_S(S: np.ndarray, T_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reconstruct an approximation of T from S by padding S with zeros.

    Parameters:
        S (np.ndarray): Tensor of shape (a_1 - 1, a_2 - 1, ..., a_k - 1)
        T_shape (tuple): Shape of T (a_1, a_2, ..., a_k)

    Returns:
        np.ndarray: Reconstructed tensor of shape T_shape
    """
    pad_widths = []
    for t_dim, s_dim in zip(T_shape, S.shape):
        pad_total = t_dim - s_dim
        if pad_total < 0:
            raise ValueError("S has larger dimensions than T in at least one axis.")
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        pad_widths.append((pad_before, pad_after))
    
    # If T has more dimensions than S, pad the additional dimensions with zeros
    if len(T_shape) > len(S.shape):
        for _ in range(len(T_shape) - len(S.shape)):
            pad_widths.append((0, 0))  # No padding for extra dimensions

    T_hat = np.pad(S, pad_width=pad_widths, mode='constant', constant_values=0)
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
    losses_S = []
    losses_boundary = []
    
    for trial in tqdm(range(num_trials), desc=f"Tensor Shape {tensor_shape}", leave=False):
        seed_T = trial + 12345  # Unique seed for T to ensure reproducibility
        
        # Generate T with real values from standard normal distribution
        T = random_real_tensor(shape=tensor_shape, mean=mean, std=std, seed=seed_T)
        
        # Compute the boundary tensor ∂T
        partial_T = bdry(T)
        S_shape = partial_T.shape

        # Generate S using the global RNG without specifying a seed to ensure independence
        S = random_real_tensor(shape=S_shape, mean=mean, std=std, seed=___SEED___)  # Uses global_rng

        # Compute losses
        loss_S = compute_loss(T, S)
        loss_boundary = compute_loss(T, partial_T)
        
        # Append the computed losses to their respective lists
        losses_S.append(loss_S)
        losses_boundary.append(loss_boundary)

    # Convert lists to numpy arrays for efficient computation
    losses_S = np.array(losses_S)
    losses_boundary = np.array(losses_boundary)

    # Calculate statistical metrics
    avg_loss_S = np.mean(losses_S)
    avg_loss_boundary = np.mean(losses_boundary)
    inequality_holds = np.sum(losses_S >= losses_boundary)
    std_loss_S = np.std(losses_S, ddof=1)
    std_loss_boundary = np.std(losses_boundary, ddof=1)
    t_statistic, p_value = stats.ttest_rel(losses_S, losses_boundary)
    mean_diff = avg_loss_S - avg_loss_boundary
    pooled_std = np.sqrt((std_loss_S ** 2 + std_loss_boundary ** 2) / 2)
    cohen_d = mean_diff / pooled_std if pooled_std != 0 else np.inf

    # Compile results into a dictionary
    results = {
        'Tensor Shape': tensor_shape,
        'Average Loss T & S': avg_loss_S,
        'Average Loss T & ∂T': avg_loss_boundary,
        'Inequality Holds (%)': (inequality_holds / num_trials) * 100,
        'Standard Deviation Loss S': std_loss_S,
        'Standard Deviation Loss ∂T': std_loss_boundary,
        'Paired t-test Statistic': t_statistic,
        'p-value': p_value,
        'Cohen\'s d': cohen_d
    }
    return results

def plot_results(csv_file: str):
    """
    Generate and save plots based on the experiment results stored in a CSV file.

    Parameters:
        csv_file (str): Path to the CSV file containing experiment results.
    """
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"CSV file '{csv_file}' not found. Skipping plotting.")
        return

    # Load the results
    df = pd.read_csv(csv_file)

    # Ensure that 'Tensor Shape' is treated as a categorical variable
    df['Tensor Shape'] = df['Tensor Shape'].astype(str)

    # Set plot style
    sns.set(style="whitegrid")

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
    plt.savefig("average_loss_comparison.png")
    plt.show()

    # ----------------------------
    # Plot 2: Inequality Holding Percentage
    # ----------------------------
    plt.figure(figsize=(14, 7))
    sns.barplot(x="Tensor Shape", y="Inequality Holds (%)", data=df, palette="viridis")
    plt.title("Percentage of Trials Where Loss(T, S) ≥ Loss(T, ∂T)")
    plt.xlabel("Tensor Shape")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("inequality_holds_percentage.png")
    plt.show()

    # ----------------------------
    # Plot 3: Paired t-test Statistic and Cohen's d
    # ----------------------------
    plt.figure(figsize=(14, 7))
    df_plot = df[['Tensor Shape', 'Paired t-test Statistic', 'Cohen\'s d']]
    df_plot = df_plot.set_index('Tensor Shape').T
    sns.heatmap(df_plot, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Paired t-test Statistic and Cohen's d Across Tensor Shapes")
    plt.xlabel("Tensor Shape")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("statistical_metrics_heatmap.png")
    plt.show()

    # ----------------------------
    # Plot 4: Distribution of Inequality Holds
    # ----------------------------
    plt.figure(figsize=(14, 7))
    sns.barplot(x="Tensor Shape", y="Inequality Holds (%)", data=df, palette="viridis")
    plt.title("Distribution of Inequality Holds Across Tensor Shapes")
    plt.xlabel("Tensor Shape")
    plt.ylabel("Inequality Holds (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("inequality_holds_distribution.png")
    plt.show()

    print("All plots have been generated and saved.")

def main():
    """
    Main function to execute the experiments, save results, and generate plots.
    """
    # Define the tensor shapes to test (both isotropic and anisotropic)
    tensor_shapes = [
        (5, 5, 5),
        (5, 5, 7),
        (5, 7, 9),
        (7, 7, 7),
        (7, 7, 7, 7),
        (9,9,9)
    ]

    num_trials = 10000  # Number of trials per tensor shape
    mean = 0.0          # Mean of the normal distribution
    std = 1.0           # Standard deviation of the normal distribution

    all_results = []  # List to store results for all tensor shapes

    for shape in tensor_shapes:
        print(f"Running experiment for tensor shape: {shape}")
        result = run_experiment(tensor_shape=shape, num_trials=num_trials, mean=mean, std=std)
        all_results.append(result)
        
        # Display the results for the current tensor shape
        print(f"Results for tensor shape {shape}:")
        print(f"  Average Loss between T and S: {result['Average Loss T & S']:.6f}")
        print(f"  Average Loss between T and ∂T: {result['Average Loss T & ∂T']:.6f}")
        print(f"  Inequality Loss(T, S) ≥ Loss(T, ∂T) holds in {result['Inequality Holds (%)']:.2f}% of trials")
        print(f"  Standard Deviation of Loss(T, S): {result['Standard Deviation Loss S']:.6f}")
        print(f"  Standard Deviation of Loss(∂T): {result['Standard Deviation Loss ∂T']:.6f}")
        print(f"  Paired t-test Statistic: {result['Paired t-test Statistic']:.6f}")
        print(f"  p-value: {result['p-value']}")
        print(f"  Cohen's d (Effect Size): {result['Cohen\'s d']:.6f}")
        print("-" * 60)

    # Save all_results to a CSV file for further analysis
    csv_file = 'experiment_results.csv'
    csv_columns = ['Tensor Shape', 'Average Loss T & S', 'Average Loss T & ∂T',
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
        print(f"All results have been saved to {csv_file}")
    except IOError:
        print("I/O error while trying to write the CSV file.")

    # Generate plots based on the saved CSV
    plot_results(csv_file)

if __name__ == "__main__":
    main()
