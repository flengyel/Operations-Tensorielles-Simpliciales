# Simplicial Tensor Operations

**Research playground for simplicial operations on matrices and hypermatrices.**

## Mathematical formalities

This project focuses on simplicial operations on matrices and hypermatrices, in particular the face maps (`d_i`), degeneracies (`s_i`), and boundary operators. The operations studied here satisfy the standard simplicial identities.

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

## Publishing your changes to GitHub

When you work inside a remote environment (for example this Codex container), commits stay local to that environment until you push them to GitHub. For this project the destination repository should be named **`SimplicialTensors`**.

1. Open the integrated terminal (see [`docs/no_terminal_troubleshooting.md`](docs/no_terminal_troubleshooting.md) if the terminal tab is hidden) and make sure you are at the workspace root:

   ```bash
   cd /workspace/Operations-Tensorielles-Simpliciales
   ```

2. Configure or update the remote so it points at your GitHub project:

   ```bash
   git remote add origin git@github.com:<your-username>/SimplicialTensors.git
   # If origin already exists, run instead:
   # git remote set-url origin git@github.com:<your-username>/SimplicialTensors.git
   ```

3. Push the active branch (the workspace uses `work` by default):

   ```bash
   git push -u origin work
   ```

4. Open `https://github.com/<your-username>/SimplicialTensors/pulls` and create a Pull Request if desired.

## Step-by-step: from this workspace to your computer

If you started in this hosted workspace and you do **not** yet have the project locally, follow this checklist. It mirrors the detailed guide in [`docs/pushing_from_codex_workspace.md`](docs/pushing_from_codex_workspace.md).

1. **Push the code from here to GitHub.**
   * Open the Terminal tab (or follow the troubleshooting guide linked above) so you can run shell commands.
   * Stay in `/workspace/Operations-Tensorielles-Simpliciales`.
   * Verify you are on the correct branch (`git status -sb`).
   * Ensure the remote targets `git@github.com:<your-username>/SimplicialTensors.git`.
   * Push the branch (`git push -u origin work`).
2. **Clone or pull on your own computer.**
   * First time: `git clone git@github.com:<your-username>/SimplicialTensors.git`.
   * Later: `git pull --ff-only` from inside the `SimplicialTensors` directory.
3. **Work with the package.**
   * Change into `SimplicialTensors/` and install with `pip install -e .`.
   * Run examples or tests as shown earlier in the README.

Cloning from GitHub automatically gives you a local folder named `SimplicialTensors`, so no manual renaming or downloading is required.
