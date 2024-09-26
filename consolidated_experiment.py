
import numpy as np
from scipy import stats
import logging
from typing import Tuple
from scipy.stats import truncnorm

# Set up logging
logging.basicConfig(level=logging.WARNING)

# ----------------------------
# Tensor Construction Functions
# ----------------------------

def random_tensor(shape: Tuple[int], low: int = 1, high: int = 10) -> np.ndarray:
    return np.random.randint(low, high, size=shape)

def random_real_tensor(shape: Tuple[int], mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    return np.random.normal(loc=mean, scale=std, size=shape)

# ----------------------------
# Boundary, Face, and Degeneracy Operations
# ----------------------------

def s_dim(t: np.ndarray) -> int:
    return min(t.shape) - 1

def degree(t: np.ndarray) -> int:
    return len(t.shape)

def _dims(m: np.ndarray) -> Tuple[np.ndarray]:
    return tuple([np.arange(dim_size) for dim_size in m.shape])

def _face(m: np.ndarray, axes: Tuple[np.ndarray], i: int) -> np.ndarray:
    indices = [np.delete(axis, i) if len(axis) > i else axis for axis in axes]
    grid = np.ix_(*indices)
    return m[grid]

def face(m: np.ndarray, i: int) -> np.ndarray:
    axes = _dims(m)
    return _face(m, axes, i)

def degen(z: np.ndarray, k: int) -> np.ndarray:
    for axis in range(z.ndim):
        slices = [slice(None)] * z.ndim
        slices[axis] = k
        z = np.insert(z, k, z[tuple(slices)], axis=axis)
    return z

def bdry(m: np.ndarray) -> np.ndarray:
    d = np.min(m.shape)
    axes = _dims(m)
    a = np.zeros(np.subtract(m.shape, np.array([1])))
    for i in range(d):
       if i % 2 == 0:
           a = np.add(a, _face(m, axes, i))
       else:
           a = np.subtract(a, _face(m, axes, i))
    return a

# ----------------------------
# Penalty Adjustment and Loss Computation
# ----------------------------

def compute_penalty(S: np.ndarray) -> float:
    return 1 + np.linalg.norm(bdry(S))

def compute_loss(T: np.ndarray, S: np.ndarray) -> float:
    T_norm = T / np.linalg.norm(T) if np.linalg.norm(T) != 0 else T
    S_norm = S / np.linalg.norm(S) if np.linalg.norm(S) != 0 else S
    return np.linalg.norm(T_norm - S_norm)

# ----------------------------
# Weight Initialization Methods
# ----------------------------

def glorot_init(shape):
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)

def he_init(shape):
    fan_in = shape[0]
    stddev = np.sqrt(2 / fan_in)
    return np.random.normal(0, stddev, size=shape)

def orthogonal_init(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0, 1, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    orthogonal_matrix = u if u.shape == flat_shape else v
    return orthogonal_matrix.reshape(shape)

def initialize_weights(shape, method='glorot'):
    if method == 'glorot':
        return glorot_init(shape)
    elif method == 'he':
        return he_init(shape)
    elif method == 'orthogonal':
        return orthogonal_init(shape)
    else:
        raise ValueError(f"Unknown initialization method: {method}")

# ----------------------------
# Experiment Setup
# ----------------------------

def permutation_test(losses_S_adjusted, losses_boundary):
    observed_diff = np.mean(losses_S_adjusted) - np.mean(losses_boundary)
    return observed_diff, 0.0  # Placeholder for p-value

def ks_test(losses_S_adjusted, losses_boundary):
    return stats.ks_2samp(losses_S_adjusted, losses_boundary)

def calculate_effect_size(losses_S_adjusted, losses_boundary):
    mean_diff = np.mean(losses_S_adjusted) - np.mean(losses_boundary)
    pooled_std = np.sqrt((np.std(losses_S_adjusted) ** 2 + np.std(losses_boundary) ** 2) / 2)
    return mean_diff / pooled_std if pooled_std != 0 else np.inf

def run_experiment_with_initialization(tensor_shape: Tuple[int, ...], init_method: str, num_trials: int = 100) -> dict:
    logging.info(f"Running experiment with {init_method} initialization for shape {tensor_shape}")

    losses_S_adjusted = []
    losses_boundary = []

    for trial in range(num_trials):
        T = initialize_weights(tensor_shape, method=init_method)
        bdry_T = bdry(T)
        S = initialize_weights(bdry_T.shape, method=init_method)
        S_reconstructed = degen(S, 0)
        penalty = compute_penalty(S)
        loss_S_adjusted = compute_loss(T, S_reconstructed) * penalty
        loss_boundary = compute_loss(T, degen(bdry_T, 0))

        losses_S_adjusted.append(loss_S_adjusted)
        losses_boundary.append(loss_boundary)

    observed_diff, perm_p_value = permutation_test(losses_S_adjusted, losses_boundary)
    ks_stat, ks_p_value = ks_test(losses_S_adjusted, losses_boundary)
    cohen_d = calculate_effect_size(np.array(losses_S_adjusted), np.array(losses_boundary))

    results = {
        "Tensor Shape": tensor_shape,
        "Initialization Method": init_method,
        "Permutation Test": {"Observed Difference": observed_diff, "p-value": perm_p_value},
        "KS Test": {"KS Statistic": ks_stat, "p-value": ks_p_value},
        "Cohen's d": cohen_d
    }

    return results

# ----------------------------
# Main Experiment
# ----------------------------

initialization_methods = ['glorot', 'he', 'orthogonal']
selected_tensor_shapes_final = [(128, 64), (32, 3, 3, 3)]

def main():
    penalty_adjusted_results_with_initializations = []

    for method in initialization_methods:
        for shape in selected_tensor_shapes_final:
            result = run_experiment_with_initialization(shape, method, num_trials=200)
            penalty_adjusted_results_with_initializations.append(result)
    
    return penalty_adjusted_results_with_initializations

if __name__ == "__main__":
    results = main()
    for res in results:
        print(res)
