# ZNEUS 2026 · Neural Networks labs @ FIIT STU

Hands-on lab materials for **ZNEUS** at the Faculty of Informatics and
Information Technologies, Slovak University of Technology in Bratislava, winter semester 2026/27.

Eight weekly labs take you from a perceptron written in raw tensors to training a CNN from scratch, followed by a
four-week project. Weeks 2–4 are practice for a test, with no notebook, commit-link or W&B-link hand-in.
Follow the stated submission requirements for the other labs and the project.

> **For labs that require a hand-in:** submit the code (a commit in your fork) and, when a model is trained,
> a public Weights & Biases link. Kaggle weeks additionally require a submission on the leaderboard.
> Weeks 2–4 are assessed by a test instead.

## Quick start

```bash
# 1. Fork this repository on GitHub (button top right), then clone YOUR fork:
git clone git@github.com:<your-github-username>/zneus-2026.git
cd zneus-2026
git remote add upstream https://github.com/fiit-ba/zneus-2026.git

# 2. Create the environment (installs Python 3.12 + PyTorch for CPU / Apple Silicon):
uv sync
#    NVIDIA GPU? Use one of these instead:
#    uv sync --no-group cpu --group cu130     # recent drivers (CUDA 13)
#    uv sync --no-group cpu --group cu126     # older drivers (CUDA 12.6)

# 3. Open the notebooks:
uv run jupyter lab
```

No `uv` yet? Never heard of it? Start with [README.md](labs/week_01_setup/README.md), it walks through everything (git
fork, uv, PyTorch on CPU/GPU, Google Colab, Kaggle, Weights & Biases). No GPU? A free Google Colab T4 is enough for
every week; [README.md](labs/week_01_setup/README.md) shows how.

## Labs

| Week | Folder | Notebooks |
|:---:|---|---|
| 1 | [week_01_setup](labs/week_01_setup/) | `01_environment_check.ipynb`, `02_torch_basics.ipynb` (optional) |
| 2 | [week_02_mlp_forward](labs/week_02_mlp_forward/) | `task_1_perceptron_and_activations.ipynb` |
| 3 | [week_03_backprop](labs/week_03_backprop/) | `task_2_backprop.ipynb` |
| 4 | [week_04_optimizers](labs/week_04_optimizers/) | `task_3_optimizers.ipynb` |
| 5 | [week_05_california_housing](labs/week_05_california_housing/) | `task_4_california_housing_mlp.ipynb` |
| 6 | [week_06_mnist](labs/week_06_mnist/) | `task_5_mnist_mlp.ipynb` |
| 7 | [week_07_overfitting](labs/week_07_overfitting/) | `task_6_fashion_mnist_2k.ipynb` |
| 8 | [week_08_cifar10_cnn](labs/week_08_cifar10_cnn/) | `task_7_cifar10_cnn.ipynb` |

## Working in Google Colab

Every notebook has an *Open in Colab* badge. The badge opens the copy in this repository; to open the copy in your
fork, replace `fiit-ba` in the URL with your GitHub username. Switch the runtime to a GPU (Runtime → Change runtime
type → T4 GPU) for week 8, run the *Colab setup* cell, and save your work back to your fork with
File → Save a copy in GitHub. 

## Datasets

Weeks 5 to 8 use the files of the week's Kaggle competition: download them from the competition's *Data* tab into
`labs/week_XX_*/data/kaggle/` (git-ignored). Without them the notebooks fall back to the public copies of the datasets
(downloaded into `data/` next to the notebook: California housing via scikit-learn ~1 MB, MNIST ~12 MB,
Fashion-MNIST ~30 MB, CIFAR-10 ~170 MB), so everything runs, but the fallback test ids are
not the Kaggle ids.
Set the environment variable `ZNEUS_DATA_DIR` to share one data folder between notebooks.

## Debugging your training

Before you spend an hour waiting for a run, read Andrej Karpathy's
[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/). The short version: look at
your data, get a tiny model to overfit a tiny subset first (`SUBSET = 256` in the configuration cell), then scale up.
Log everything to W&B so a crashed session still leaves the curves behind.


## TODO (instructors): Kaggle competitions

The [week 1 warm-up invitation](https://www.kaggle.com/t/5e8b5441c9144adbaa973dceac384939) and [upload instructions](labs/week_01_setup/README.md) are available here.
Four later competitions still need to be created before their labs; their links will be added when ready. Each item says what the notebook expects in the competition's *Data* tab and what the students'
`submission.csv` looks like.

- [x] **Week 1 warm-up.** Launched with invitation-only joining. Data: `test.csv` (`id`) and
  `sample_submission.csv` (`id`, `prediction`), 1,000 IDs `0..999`. Metric: RMSE against `id / 999`.
  Deadline: 30 September 2026, 23:59 Europe/Bratislava. Successful processing is the warm-up completion criterion.
- [ ] **Week 5 California housing.** Data: `train.csv` (features + `MedHouseVal`), `test.csv` (`id` + features),
  `sample_submission.csv`. Submission: `id`, `MedHouseVal`.
- [ ] **Week 6 MNIST.** Data: `train.npz` with arrays `x`, `y`; `test.npz` with `x`. Submission: `id` (row index
  `0..n-1`), `label`.
- [ ] **Week 7 Fashion-MNIST 2k.** Data: `train.npz` (`x`, `y`, 2 000 images), `test.npz` (`x`). Submission: `id`, `label`.
- [ ] **Week 8 CIFAR-10.** Data: `train.npz` (`x`, `y`), `test.npz` (`x`), about 170 MB. Submission: `id`, `label`.

For every competition, set the evaluation metric to the one the week's README names.


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
