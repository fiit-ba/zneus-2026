# Setup

## Before you start

- **git**: [install it](https://git-scm.com/downloads), then check with `git --version`.
- A **GitHub account**: [github.com/signup](https://github.com/signup). Your fork is where every hand-in lives.
- About **5 GB of free disk space**.
- **No Python needed.** uv downloads the right version and keeps it inside the project folder.
- **Windows**: use PowerShell or Git Bash. WSL2 works too and behaves like Linux.

## Fork and clone

1. Open [github.com/fiit-ba/zneus-2026](https://github.com/fiit-ba/zneus-2026) and click **Fork** (top right). Keep
   the name `zneus-2026`.
2. Clone **your fork** and register ours as `upstream`:

   ```bash
   git clone https://github.com/<your-github-username>/zneus-2026.git
   cd zneus-2026
   git remote add upstream https://github.com/fiit-ba/zneus-2026.git
   ```

3. Check: `git remote -v` lists `origin` (your fork) and `upstream` (fiit-ba).

The first time you push, GitHub asks for a
[personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
instead of a password. The [GitHub CLI](https://cli.github.com/) (`gh auth login`) sets this up for you.

## Install uv

[uv](https://docs.astral.sh/uv/) installs Python, creates the environment and installs the packages. It replaces
`pip` and `venv`. Everyone installs from the same lock file, so everyone runs the same versions.

```bash
# macOS, Linux, WSL2
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Close and reopen the terminal**, then check with `uv --version`.

## Create the environment

```bash
uv sync
```

This downloads Python 3.12, creates `.venv/` and installs every package (about 1 GB the first time, seconds
afterwards). You get the **CPU build of PyTorch**, which on an Apple Silicon Mac also uses the GPU through MPS.
NVIDIA GPU: see the [next section](#nvidia-gpu).

Check:

```bash
uv run python -c "import torch; print(torch.__version__)"
```

**Running things.** Put `uv run` in front of a command and it runs inside `.venv/`. No activation needed:

```bash
uv run jupyter lab
uv run wandb login
```

If you prefer the classic way, activate the environment once per terminal:

```bash
source .venv/bin/activate        # macOS, Linux, WSL2, Git Bash
.venv\Scripts\activate           # Windows PowerShell
```

**Adding a package:** `uv add <package>`. Do not edit `pyproject.toml` by hand.

## NVIDIA GPU

Skip this section if you have no NVIDIA GPU. AMD and Intel GPUs are not supported; use the CPU build or
[Colab](#google-colab).

1. Run `nvidia-smi`. The top right corner shows `CUDA Version: 13.0` or `12.x`.
   If the command is not found, install the driver from [nvidia.com/drivers](https://www.nvidia.com/drivers) and
   reboot. You do not need the CUDA toolkit.
2. Pick the matching command:

   | `nvidia-smi` says | Command |
   |---|---|
   | CUDA Version 13.x | `uv sync --no-group cpu --group cu130` |
   | CUDA Version 12.x | `uv sync --no-group cpu --group cu126` |

   The download is several GB the first time.
3. Check:

   ```bash
   uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   # 2.14.0+cu130 True
   ```

**Keep the GPU build.** A plain `uv sync` or `uv run` switches you **back to the CPU build**. Once you are on the
GPU build, always add `--no-sync`:

```bash
uv run --no-sync jupyter lab
```

Or activate the environment and run `jupyter lab` directly. Update the environment only with the full GPU command
from the table. If you keep forgetting, put `export UV_NO_SYNC=1` in your shell profile.

## Open the notebooks

**JupyterLab:**

```bash
uv run jupyter lab               # GPU build: uv run --no-sync jupyter lab
```

A browser tab opens. The notebooks are under `labs/`. Stop the server with Ctrl+C.

**VS Code** (or Cursor):

1. Install the **Python** and **Jupyter** extensions.
2. *File* → *Open Folder* → the `zneus-2026` folder (the folder, not a single file).
3. Open a notebook, click *Select Kernel* (top right) → *Python Environments* → `.venv`.

After every `uv sync`, restart the kernel (*Kernel* → *Restart Kernel*), otherwise it keeps the old packages.

Now run [`01_environment_check.ipynb`](01_environment_check.ipynb). It checks everything on this page.

## Google Colab

[Google Colab](https://colab.research.google.com) runs notebooks on Google's machines with a free NVIDIA T4 GPU. Use
it when your laptop has no GPU (weeks 7 and 8, and the project) or when your own setup misbehaves. You only need a
Google account.

1. **Open a notebook.** Click the *Open in Colab* badge at the top of the notebook. It opens the copy in **our**
   repository; to open the copy in **your fork**, replace `fiit-ba` in the URL with your GitHub username.
2. **Turn on the GPU.** *Runtime* → *Change runtime type* → *T4 GPU* → *Save*.
3. **Run the first cell** (Colab setup). It installs what Colab is missing and does nothing on your own machine.
4. **Save your work back to your fork.** *File* → *Save a copy in GitHub* → repository
   `<your-github-username>/zneus-2026`, branch `main`, same file path. Colab creates a commit in your fork. That
   commit is your hand-in.

Good to know:

- Files on the Colab machine are **gone when the session ends**. Re-run the download cell next time. Download
  `submission.csv` from the *Files* pane on the left before you close the tab.
- Sessions disconnect after some inactivity and last at most about 12 hours. Log to W&B, so a lost session still
  leaves the curves behind.
- To keep files between sessions, mount your Google Drive:

  ```python
  from google.colab import drive
  drive.mount("/content/drive")      # then save to /content/drive/MyDrive/zneus-2026/...
  ```

- `wandb.init()` asks for your W&B API key; paste it from [wandb.ai/authorize](https://wandb.ai/authorize).

## Kaggle

[Kaggle](https://www.kaggle.com) hosts the leaderboards of weeks 1 and 4 to 8. You upload a `submission.csv` with
predictions and the leaderboard scores it.

1. **Account.** Sign up at [kaggle.com](https://www.kaggle.com/account/login). Then **verify your phone number** in
   *Settings*. Without it you cannot submit.
2. **Name yourself.** Set your **display name** to your **AIS login** (profile → *Edit profile*). If a competition
   has a *Team* tab, set the **team name** to your AIS login too. Unnamed entries cannot be graded.
3. **Join the competition** through the link given at the lab (**Join Competition**, accept the rules).
4. **Download the data** from the competition's *Data* tab into `labs/week_XX_*/data/kaggle/`. Without it the
   notebook falls back to a public copy of the dataset and your submission does not score.
5. **Submit.** Click **Submit Prediction**, upload the `submission.csv` the notebook wrote, add a short description.
   Your score appears under *My Submissions* and on the *Leaderboard*.

Prefer the terminal? Log in once, then submit:

```bash
uv run kaggle auth login           # opens the browser, click Allow
uv run kaggle competitions submit -c <competition-slug> -f submission.csv -m "week 4 mlp, 3 layers"
```

The slug is the last part of the competition URL. Submissions per day are limited, so do not upload every epoch. The
public leaderboard uses only part of the test set; the final score uses the rest. Trust your own validation split.

## Weights and Biases

[Weights & Biases](https://wandb.ai) (W&B) records your training runs and gives you a link with the curves. **Every
hand-in that trains a model needs a public W&B link**, so set it up now.

1. **Account.** Sign up at [wandb.ai](https://wandb.ai/site) with your `@stuba.sk` e-mail.
2. **Log in** once per machine and paste the key from [wandb.ai/authorize](https://wandb.ai/authorize):

   ```bash
   uv run wandb login
   ```

   In Colab, the notebook asks for the key when the cell runs. Never paste the key into a notebook.
3. **Run a notebook.** They all log to the project `zneus-2026`, which appears on your account after the first run.
4. **Make the project public.** Open `https://wandb.ai/<your-username>/zneus-2026`, click the **lock icon** next
   to the project name, set the visibility to **Public**, *Save*. Once is enough. If we cannot open your link, the
   run does not count.
5. **Hand in** the project URL or the URL of a specific run (printed by `wandb.init`).

No account yet, or offline? Start Jupyter with `WANDB_MODE=offline uv run jupyter lab`. Runs go to a local `wandb/`
folder; upload them later with `uv run wandb sync wandb/offline-run-*`.

## New weeks and hand-ins

**Before each lab**, pull the new material from our repository:

```bash
git fetch upstream
git merge upstream/main
uv sync                            # GPU build: uv sync --no-group cpu --group cu130
```

**After each lab**, commit the week's folder and push it to your fork:

```bash
git add labs/week_02_mlp_forward
git commit -m "week 2: perceptron and MLP forward pass"
git push
```

Your hand-in link is the commit (`https://github.com/<your-github-username>/zneus-2026/commit/<sha>`) or the week's
folder in your fork. Notebook outputs are welcome. Never commit `data/`, `wandb/`, `.venv/`, `kaggle.json` or
`submission.csv`; they are git-ignored already.

**Merge conflict?** `git status` lists the files. Keep your version with `git checkout --ours <path>`, or take ours
with `git checkout --theirs <path>`, then `git add <path>` and `git commit`. If `uv.lock` conflicts, take ours.
