# Opening a shell in the Codex workspace

Some Codex sessions are launched without the terminal tab visible by default. If you do not see a **Terminal** entry anywhere in the UI, use the steps below to create one.

1. Move the mouse to the upper-left corner of the window and click the **hamburger menu** (three horizontal lines).
2. Choose **Terminal** → **New Terminal**. The first click may take a few seconds to open a new panel.
3. If the menu is collapsed, press **Ctrl+Shift+P** (or **Cmd+Shift+P** on macOS) to open the command palette, then type `Terminal: Create New Terminal` and press Enter.
4. The terminal panel should appear at the bottom with a prompt similar to:
   ```
   root@<container-id>:/workspace/Operations-Tensorielles-Simpliciales#
   ```
5. If nothing appears after several seconds, click the **+** button in the terminal panel or repeat step 3. Slow network connections can delay the first shell startup.

## Verifying you are in the repository

Once a shell opens, the prompt should end with `/workspace/Operations-Tensorielles-Simpliciales`. If it does not, run:

```bash
pwd
cd /workspace/Operations-Tensorielles-Simpliciales
```

From there you can run `git status`, `pytest -q`, or any other commands described in the documentation.

## Still stuck?

If the workspace refuses to open a terminal, refresh the browser tab. As a last resort, close the session from the Codex dashboard and start a new one; each session automatically checks out the repository into `/workspace/Operations-Tensorielles-Simpliciales`.
