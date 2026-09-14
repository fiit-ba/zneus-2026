# ZNEUS 2026 · Neural Networks labs @ FIIT STU

Hands-on lab materials for **ZNEUS** at the Faculty of Informatics and
Information Technologies, Slovak University of Technology in Bratislava, winter semester 2026/27.

Eight weekly labs take you from a perceptron written in raw tensors to training a CNN from scratch, followed by a
four-week project. Weeks 2–4 are practice for a test, with no notebook, commit-link or W&B-link hand-in.
Follow the stated submission requirements for the other labs and the project.

**Materials are released week by week.** The table below shows what is available. At the start of each lab,
[update your private repository](labs/week_01_setup/README.md#new-weeks-and-hand-ins) to get the newly released folder.

> - **Week 1:** one successfully scored [Kaggle warm-up submission](labs/week_01_setup/README.md#kaggle).
> - **Weeks 2–4:** practice for a test; no notebook or link submission.
> - **Weeks 5–8:** code committed and pushed to your private repository, plus a Kaggle submission. Paste the commit link
>   into the **description of your Kaggle submission**: that box is how your links reach us.
> - **Projects:** log your experiments to Weights & Biases and include a public W&B project link with your submission.
>
> W&B logging is optional for labs; no lab requires a W&B link.

## Repository privacy

**Keep GitHub repositories containing your lab and project solutions private and invite your lab instructor(s)
as collaborators.** Solutions and feedback can contain students' personal data; access for assessment should be
limited to the teaching staff involved. Make sure your instructors have accepted the invitation and can open
your submitted commit links.

Create a separate private copy using the [setup instructions](labs/week_01_setup/README.md#create-your-private-repository).
[Forks of public GitHub repositories are public](https://docs.github.com/en/pull-requests/reference/forks#visibility-of-forks),
so follow the steps below to create a private copy.

### Instructor GitHub accounts

Use these usernames when inviting your lab instructor(s) as collaborators:

- [@MarkoStahovec](https://github.com/MarkoStahovec)
- [@williambrach](https://github.com/williambrach)

## Quick start

On GitHub, [create an empty repository](https://github.com/new) named `zneus-2026` with visibility **Private**.
Leave the README, `.gitignore` and licence options unchecked. Replace `<your-github-username>` below with your username.
Already have a public fork? Follow the [migration steps](labs/week_01_setup/README.md#already-have-a-public-fork).

```bash
# 1. Copy the course materials into your private repository:
git clone https://github.com/fiit-ba/zneus-2026.git
cd zneus-2026
git remote rename origin upstream
git remote add origin https://github.com/<your-github-username>/zneus-2026.git
git push -u origin main

# 2. Create the environment (installs Python 3.12 + PyTorch for CPU / Apple Silicon):
uv sync
#    NVIDIA GPU? Use one of these instead:
#    uv sync --no-group cpu --group cu130     # recent drivers (CUDA 13)
#    uv sync --no-group cpu --group cu126     # older drivers (CUDA 12.6)

# 3. Open the notebooks:
uv run jupyter lab
```

In your private repository, open **Settings → Collaborators → Add people** and invite your lab instructor(s)
using the [instructor GitHub accounts](#instructor-github-accounts) above.

No `uv` yet? Never heard of it? Start with [README.md](labs/week_01_setup/README.md), it walks through everything (git
setup, uv, PyTorch on CPU/GPU, Google Colab, Kaggle, Weights & Biases). No GPU? A free Google Colab T4 is enough for
every week; [README.md](labs/week_01_setup/README.md) shows how.

## Labs

| Week | Material | Availability |
|:---:|---|---|
| 1 | [Environment setup](labs/week_01_setup/) | Available: `01_environment_check.ipynb`, `02_torch_basics.ipynb` (optional) |
| 2 | [MLP forward pass](labs/week_02_mlp_forward/) | Available: `task_1_perceptron_and_activations.ipynb` |
| 3 | Backpropagation | Released in week 3 |
| 4 | Optimizers | Released in week 4 |
| 5 | California housing | Released in week 5 |
| 6 | MNIST | Released in week 6 |
| 7 | Overfitting | Released in week 7 |
| 8 | CIFAR-10 CNN | Released in week 8 |
| Project | Topics and requirements | Released after the labs |

## Working in Google Colab

Every notebook has an *Open in Colab* badge that opens the course copy. To open your private copy, use
[Colab's GitHub browser](https://colab.research.google.com/github), enable **Include Private Repos**, authorise
GitHub access and select your repository and notebook. Switch the runtime to a GPU (Runtime → Change runtime
type → T4 GPU) for week 8, run the *Colab setup* cell, and save your work to your private repository with
File → Save a copy in GitHub. See the [Colab setup instructions](labs/week_01_setup/README.md#google-colab).

## Debugging your training

Before you spend an hour waiting for a run, read Andrej Karpathy's
[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/). The short version: look at
your data, get a tiny model to overfit a tiny subset first (`SUBSET = 256` in the configuration cell), then scale up.
You can use W&B to keep and compare training curves during labs. Experiment tracking with W&B is required for projects.


## General resources

**Maths refreshers**
- [Linear algebra and calculus refresher](https://stanford.edu/~shervine/teaching/cs-229/refresher-algebra-calculus/) (Shervine Amidi)
- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/) (Parr & Howard)
- [3Blue1Brown: Essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra)

**Neural networks**
- [Dive into Deep Learning](https://d2l.ai/) (free book with PyTorch code, our main reference)
- [Deep Learning](https://www.deeplearningbook.org/) (Goodfellow, Bengio, Courville)
- [3Blue1Brown: Neural networks](https://www.3blue1brown.com/topics/neural-networks)
- [Andrej Karpathy: Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Stanford CS231n notes](https://cs231n.github.io/)

**PyTorch**
- [PyTorch tutorials: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) (Daniel Bourke)

**Tools**
- [uv documentation](https://docs.astral.sh/uv/)
- [Weights & Biases quickstart](https://docs.wandb.ai/quickstart/)
- [Kaggle: how to submit to a competition](https://www.kaggle.com/docs/competitions)
- [git – the simple guide](https://rogerdudler.github.io/git-guide/)
