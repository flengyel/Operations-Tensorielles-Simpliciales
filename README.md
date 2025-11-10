# Simplicial Tensor Operations

**Simplicial operations on matrices and hypermatrices.**

## Mathematical formalities

This project defines sumplicial operations  matrices and hypermatrices (tensors), including face maps (`d_i`), degeneracies (`s_i`), and boundary operators, as well mathematical computer experiments for the diagonal sinplicial tensor module X(s; A), where s=(n_1,...,n_k) is the shape of a tensor of order k.

The operations defined here satisfy the standard simplicial identities.

$$
\begin{aligned}
d_i d_j &= d_{j-1} d_i, && \text{if } i < j; \\
s_i s_j &= s_j s_{i-1}, && \text{if } i > j; \\
d_i s_j &=
\begin{cases}
s_{j-1} d_i, & \text{if } i < j; \\
1, & \text{if } i \in \{j, j+1\}; \\
s_j d_{i-1}, & \text{if } i > j+1.
\end{cases}
\end{aligned}
$$

## Where are the sources now?

The refactor places all installable code under `src/simplicial_tensors/`. The repository root can remain `Operations-Tensorielles-Simpliciales`; during installation Python resolves the package via `simplicial_tensors` contained in `src/`. No folder renaming is required.

```
Operations-Tensorielles-Simpliciales/
├── pyproject.toml
└── src/
    └── simplicial_tensors/
        ├── __init__.py
        ├── tensor_ops.py
        └── ...
```

To work with these files on your own machine:

1. Clone or update the repository (`git clone` the first time, otherwise `git pull`).
2. Change into the `Operations-Tensorielles-Simpliciales` directory.
3. Install the package in editable mode:

   ```bash
   pip install -e .
   ```

You can then import modules via `import simplicial_tensors.tensor_ops`.

## Examples

The interactive demonstrations that used to live inside the library modules now reside in the `examples/` folder. After installing the project in editable mode (`pip install -e .`), run any example like so:

```bash
python examples/tensor_ops.py
```

Each script imports the corresponding module under `simplicial_tensors` and calls its `main()` entry point.

