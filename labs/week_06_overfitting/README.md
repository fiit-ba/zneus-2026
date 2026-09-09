# Week 6 · Something is wrong with this model

Classify 28 x 28 grayscale images of clothing into 10 categories, with a catch: you get only 2 000 training images (200 per class), while the test set has 10 000. A colleague trained an MLP on all 2 000 images: training accuracy 100 %, leaderboard score about 0.80. Beat it. Metric: accuracy.

Notebook: [`task_6_fashion_mnist_2k.ipynb`](task_6_fashion_mnist_2k.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiit-ba/zneus-2026/blob/main/labs/week_06_overfitting/task_6_fashion_mnist_2k.ipynb) (in Colab, replace `fiit-ba` in the URL with your GitHub username to open the copy in your fork).

## Data

Kaggle competition: the link is given at the lab. Download `train.npz` and `test.npz` from its *Data* tab into `labs/week_06_overfitting/data/kaggle/`.

- `train.npz`: `x`, a `uint8` array `(2000, 28, 28)`, and `y`, the labels 0..9 (200 per class). **Train only on these 2 000 images.**
- `test.npz`: `x`, 10 000 test images `(10000, 28, 28)`, no labels, shuffled; row `i` is the id `i` of your submission.
- Labels: 0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot.
- Without the Kaggle files the notebook rebuilds the same 2 000 images from the torchvision copy of Fashion-MNIST (30 MB); that submission will not score on Kaggle.

The notebook loads the files for you. Everything else, from looking at the data to the final predictions, is your work.

## Task

Train the best classifier you can on the 2 000 images, log every run to Weights & Biases (project `zneus-2026`, run names `week06-...`) and submit your predictions for `test.npz` to Kaggle (`submission.csv`, columns `id,label`, 10 000 rows). Beat 0.80 on the leaderboard.

## Hand-in

Due **by the end of the lab day (23:59)**. Every hand-in consists of the code (a commit in your fork) and, for anything that trains a model, a public Weights & Biases link; Kaggle weeks additionally require a submission on the leaderboard.

- [ ] notebook committed and pushed to your fork (commit link)
- [ ] `submission.csv` on the Kaggle leaderboard above 0.80 (team / display name = your AIS login)
- [ ] public W&B link to the project `zneus-2026` with your week 6 runs

Points for this task: ask at the lab.

## Resources

- [d2l.ai: Generalization](https://d2l.ai/chapter_linear-regression/generalization.html)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research)
- [W&B: compare runs](https://docs.wandb.ai/guides/app/features/panels/run-comparer/)
