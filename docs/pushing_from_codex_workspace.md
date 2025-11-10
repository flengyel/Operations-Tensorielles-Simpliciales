# Publishing Workspace Changes to GitHub (using this project)

This walkthrough uses the **exact repository names in this workspace** so you can copy the commands without guessing. The code lives in `/workspace/Operations-Tensorielles-Simpliciales`, and you want it to appear on GitHub under a repository named **`SimplicialTensors`**.

> **Goal:** Push the existing `work` branch from the Codex workspace to `github.com/<your-username>/SimplicialTensors`, then fetch those commits on your own computer.

## 0. Create the GitHub repository (one-time)

In a browser, sign in to GitHub and create a new empty repository named `SimplicialTensors` under your account. Do **not** initialize it with a README or any files.

## 1. Open the Codex shell at the repo root

Even though you interact with the notebook-style Codex UI, there **is** a full terminal available in the lower pane. To open it:

1. Click the **Terminal** tab in the bottom panel. If it is missing, follow the dedicated checklist in [`no_terminal_troubleshooting.md`](no_terminal_troubleshooting.md) to reveal it.
2. A shell prompt such as `root@...:/workspace#` will appear. This prompt is the same Linux shell shown in the screenshot above.

Once you can see the prompt, change into the project folder:

```bash
cd /workspace/Operations-Tensorielles-Simpliciales
```

You should now see the prompt update to `/workspace/Operations-Tensorielles-Simpliciales#`. Staying in this directory ensures every command below acts on the correct Git repository.

## 2. Configure your Git identity (once per workspace)

```bash
git config --global user.name "<your full name>"
git config --global user.email "<your-email-used-on-github>"
```

Use the same email that GitHub expects so commits show up correctly.

## 3. Point this workspace at the new GitHub repo

Add a remote that uses the real destination name:

```bash
git remote add origin git@github.com:<your-username>/SimplicialTensors.git
```

If a remote named `origin` already exists (from earlier experiments), update it instead:

```bash
git remote set-url origin git@github.com:<your-username>/SimplicialTensors.git
```

Verify the result:

```bash
git remote -v
```

You should see `origin  git@github.com:<your-username>/SimplicialTensors.git` for both fetch and push.

## 4. Test your GitHub authentication

The workspace already runs an SSH agent, but GitHub must know your SSH public key. Run:

```bash
ssh -T git@github.com
```

If GitHub greets you by username (or reports that you are already authenticated), you are ready to push. If it asks you to add the key, copy the contents of `~/.ssh/id_rsa.pub` and add it to your GitHub SSH keys page.

## 5. Push the branch named `work`

Find your current branch (it should already be `work`):

```bash
git status -sb
```

Then push it to the `SimplicialTensors` repository:

```bash
git push -u origin work
```

The `-u` flag sets `origin/work` as the default upstream so future pushes can use plain `git push`.

If Git warns about non-fast-forward updates, add `--force-with-lease` only when you are certain no one else has pushed to that branch.

## 6. Create the pull request on GitHub

Visit `https://github.com/<your-username>/SimplicialTensors/pulls`. GitHub will prompt you to open a pull request from the `work` branch you just pushed.

## 7. Bring the code onto your own computer

On your personal machine, open a terminal and run:

```bash
git clone git@github.com:<your-username>/SimplicialTensors.git
cd SimplicialTensors
```

The clone command automatically names the local directory `SimplicialTensors`, so there is no manual renaming step.

Whenever you want the latest commits from the Codex workspace, run:

```bash
git pull --ff-only
```

Now both the Codex workspace and your local machine share the exact same repository name (`SimplicialTensors`) and commit history.

