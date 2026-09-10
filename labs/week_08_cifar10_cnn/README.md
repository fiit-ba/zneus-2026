# Week 8 · CNN from scratch: CIFAR-10

Classify 32 x 32 colour photographs of 10 everyday classes with a convolutional neural network trained from scratch. Metric: accuracy; target at least 80 %.

Notebook: [`task_7_cifar10_cnn.ipynb`](task_7_cifar10_cnn.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiit-ba/zneus-2026/blob/main/labs/week_08_cifar10_cnn/task_7_cifar10_cnn.ipynb) (in Colab, replace `fiit-ba` in the URL with your GitHub username to open the copy in your fork).

**Use a GPU this week** (Colab: *Runtime -> Change runtime type -> T4 GPU*). On a laptop CPU develop with `SUBSET` set in the configuration cell and do the real runs on a GPU.

## Data

Kaggle competition: the link is given at the lab. Download `train.npz` and `test.npz` (about 170 MB) from its *Data* tab into `labs/week_08_cifar10_cnn/data/kaggle/`.

- `train.npz`: `x`, a `uint8` array `(50000, 32, 32, 3)` (height, width, RGB, values 0..255), and `y`, the labels 0..9.
- `test.npz`: `x`, 10 000 test images `(10000, 32, 32, 3)`, no labels, shuffled; row `i` is the id `i` of your submission.
- Labels: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck.
- Without the Kaggle files the notebook uses the torchvision copy of CIFAR-10 (170 MB); that submission will not score on Kaggle.

The notebook loads the files for you. Everything else, from looking at the data to the final predictions, is your work.

## Task

Train the best CNN you can from scratch (no pretrained weights), log every run to Weights & Biases (project `zneus-2026`, run names `week08-...`) and submit your predictions for `test.npz` to Kaggle (`submission.csv`, columns `id,label`, 10 000 rows). Target: at least 80 %.

## Hand-in

Due **by the end of the lab day (23:59)**. Every hand-in consists of the code (a commit in your fork) and, for anything that trains a model, a public Weights & Biases link; Kaggle weeks additionally require a submission on the leaderboard.

- [ ] notebook committed and pushed to your fork (commit link)
- [ ] `submission.csv` on the Kaggle leaderboard, at least 80 % (team / display name = your AIS login)
- [ ] public W&B link to the project `zneus-2026` with your week 8 runs

Points for this task: ask at the lab.

## Resources

- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [d2l.ai: Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html)
- [The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
