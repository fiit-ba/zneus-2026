# ZNEUS 2026 · Neural Networks labs @ FIIT STU

Hands-on lab materials for **ZNEUS** at the Faculty of Informatics and
Information Technologies, Slovak University of Technology in Bratislava, winter semester 2026/27.

Eight weekly labs take you from a perceptron written in raw tensors to fine-tuning a pretrained ResNet, followed by a
four-week project. Every lab ends with something you hand in.

> **The golden rule.** Every hand-in consists of the code (a commit in your fork) and, for anything that trains a
> model, a public Weights & Biases link. Kaggle weeks additionally require a submission on the leaderboard.
> No W&B link, no code, no points.

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

No `uv` yet? Never heard of it? Start with [SETUP.md](labs/week_01_setup/SETUP.md), it walks through everything (git
fork, uv, PyTorch on CPU/GPU, Google Colab, Kaggle, Weights & Biases). No GPU? A free Google Colab T4 is enough for
every week; SETUP.md shows how.

## Labs

| Week | Folder | Notebooks |
|:---:|---|---|
| 1 | [week_01_setup](week_01_setup/) | `01_environment_check.ipynb`, `02_torch_basics.ipynb` (optional) |
| 2 | [week_02_mlp_forward](week_02_mlp_forward/) | `task_1_perceptron_and_activations.ipynb` |
| 3 | [week_03_backprop_and_optimizers](week_03_backprop_and_optimizers/) | `task_2_backprop.ipynb`, `task_3_optimizers.ipynb` |
| 4 | [week_04_california_housing](week_04_california_housing/) | `task_4_california_housing_mlp.ipynb` |
| 5 | [week_05_mnist](week_05_mnist/) | `task_5_mnist_mlp.ipynb` |
| 6 | [week_06_overfitting](week_06_overfitting/) | `task_6_fashion_mnist_2k.ipynb` |
| 7 | [week_07_cifar10_cnn](week_07_cifar10_cnn/) | `task_7_cifar10_cnn.ipynb` |
| 8 | [week_08_transfer_learning](week_08_transfer_learning/) | `task_8_pets_transfer_learning.ipynb` |

## Working in Google Colab

Every notebook has an *Open in Colab* badge. The badge opens the copy in this repository; to open the copy in your
fork, replace `fiit-ba` in the URL with your GitHub username. Switch the runtime to a GPU (Runtime → Change runtime
type → T4 GPU) for weeks 7 and 8, run the *Colab setup* cell, and save your work back to your fork with
File → Save a copy in GitHub. 

## Datasets

Weeks 4 to 8 use the files of the week's Kaggle competition: download them from the competition's *Data* tab into
`labs/week_XX_*/data/kaggle/` (git-ignored). Without them the notebooks fall back to the public copies of the datasets
(downloaded into `data/` next to the notebook: California housing via scikit-learn ~1 MB, MNIST ~12 MB,
Fashion-MNIST ~30 MB, CIFAR-10 ~170 MB, Oxford-IIIT Pets ~800 MB), so everything runs, but the fallback test ids are
not the Kaggle ids.
Set the environment variable `ZNEUS_DATA_DIR` to share one data folder between notebooks.

## Debugging your training

Before you spend an hour waiting for a run, read Andrej Karpathy's
[A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/). The short version: look at
your data, get a tiny model to overfit a tiny subset first (`SUBSET = 256` in the configuration cell), then scale up.
Log everything to W&B so a crashed session still leaves the curves behind.


## TODO (instructors): Kaggle competitions

Six competitions have to be created before the semester starts. The links are not in the repository: students get them
at the lab. Each item says what the notebook expects in the competition's *Data* tab and what the students'
`submission.csv` looks like.

- [ ] **Week 1 warm-up.** Data: `sample_submission.csv` with columns `id`, `prediction`; without it the notebook uses
  ids `0..999`, so use that range. Submission: `id`, `prediction` (random numbers, any metric).
- [ ] **Week 4 California housing.** Data: `train.csv` (features + `MedHouseVal`), `test.csv` (`id` + features),
  `sample_submission.csv`. Submission: `id`, `MedHouseVal`.
- [ ] **Week 5 MNIST.** Data: `train.npz` with arrays `x`, `y`; `test.npz` with `x`. Submission: `id` (row index
  `0..n-1`), `label`.
- [ ] **Week 6 Fashion-MNIST 2k.** Data: `train.npz` (`x`, `y`, 2 000 images), `test.npz` (`x`). Submission: `id`, `label`.
- [ ] **Week 7 CIFAR-10.** Data: `train.npz` (`x`, `y`), `test.npz` (`x`), about 170 MB. Submission: `id`, `label`.
- [ ] **Week 8 Oxford-IIIT Pets.** Data: `train.csv` (`id`, `breed`), `sample_submission.csv` (`id`, `breed`), folders
  `train/<id>.jpg` and `test/<id>.jpg`, about 200 MB. Submission: `id`, `breed` (breed name, not index).

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
