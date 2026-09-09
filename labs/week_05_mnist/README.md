# Week 5 · MLP for images: MNIST

Classify 28 x 28 grayscale images of handwritten digits (0 to 9). Metric: accuracy.

Notebook: [`task_5_mnist_mlp.ipynb`](task_5_mnist_mlp.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiit-ba/zneus-2026/blob/main/labs/week_05_mnist/task_5_mnist_mlp.ipynb) (in Colab, replace `fiit-ba` in the URL with your GitHub username to open the copy in your fork).

## Data

Kaggle competition: the link is given at the lab. Download `train.npz` and `test.npz` from its *Data* tab into `labs/week_05_mnist/data/kaggle/`.

- `train.npz`: `x`, a `uint8` array `(60000, 28, 28)` with pixel values 0..255, and `y`, the labels 0..9. Load with `np.load("train.npz")["x"]`.
- `test.npz`: `x`, 10 000 test images `(10000, 28, 28)`, no labels, shuffled; row `i` is the id `i` of your submission.
- Without the Kaggle files the notebook uses the torchvision copy of MNIST (12 MB download); that submission will not score on Kaggle.

The notebook loads the files for you. Everything else, from looking at the data to the final predictions, is your work.

## Task

Train the best MLP you can, log every run to Weights & Biases (project `zneus-2026`, run names `week05-...`) and submit your predictions for `test.npz` to Kaggle (`submission.csv`, columns `id,label`, 10 000 rows).

## Hand-in

Due **by the end of the lab**. Every hand-in consists of the code (a commit in your fork) and, for anything that trains a model, a public Weights & Biases link; Kaggle weeks additionally require a submission on the leaderboard.

- [ ] notebook committed and pushed to your fork (commit link)
- [ ] `submission.csv` on the Kaggle leaderboard (team / display name = your AIS login)
- [ ] public W&B link to the project `zneus-2026` with your week 5 runs

Points for this task: ask at the lab.

## Resources

- [PyTorch tutorial: Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [3Blue1Brown: Neural networks](https://www.3blue1brown.com/topics/neural-networks)
- [d2l.ai: Softmax regression](https://d2l.ai/chapter_linear-classification/softmax-regression.html)
