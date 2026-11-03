# Week 8 · CNN from scratch: CIFAR-10

Classify 32 x 32 colour photographs of 10 everyday classes with a convolutional neural network trained from scratch. Metric: accuracy; target at least 80 %.

Notebook: [`task_7_cifar10_cnn.ipynb`](task_7_cifar10_cnn.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiit-ba/zneus-2026/blob/main/labs/week_08/task_7_cifar10_cnn.ipynb)

For your private copy, follow the [Colab setup instructions](../week_01_setup/README.md#google-colab).

**Use a GPU this week** (Colab: *Runtime -> Change runtime type -> T4 GPU*). On a laptop CPU develop with `SUBSET` set in the configuration cell and do the real runs on a GPU.

## Join the Kaggle competition

Open the [Week 8 invitation link](https://www.kaggle.com/t/22274cd2e419434fbea2e5e70c9769f9), sign in to Kaggle, then read and accept the competition rules to join. Use your **AIS login** as your team name.

## Data

After joining the [Kaggle competition](https://www.kaggle.com/t/22274cd2e419434fbea2e5e70c9769f9), download `train.npz` and `test.npz` (about 185 MB) from its *Data* tab into `labs/week_08/data/kaggle/`.

- `train.npz`: `x`, a `uint8` array `(50000, 32, 32, 3)` (height, width, RGB, values 0..255), and `y`, the labels 0..9.
- `test.npz`: `x`, 10 000 test images `(10000, 32, 32, 3)`, no labels, shuffled; row `i` is the id `i` of your submission.
- Labels: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck.
- Without the Kaggle files the notebook uses the torchvision copy of CIFAR-10 (170 MB); that submission will not score on Kaggle.

The notebook loads the files for you. Everything else, from looking at the data to the final predictions, is your work.

## Task

Train the best CNN you can from scratch (no pretrained weights) and submit your predictions for `test.npz` to Kaggle (`submission.csv`, columns `id,label`, 10 000 rows). Target: at least 80 %. Use only the supplied training images: external data, other copies of CIFAR-10 and looking up test labels are not permitted.

W&B logging is optional for this lab; no W&B link is required. If you do log the runs, add the link to the Kaggle submission description next to your commit link.

## Hand-in

Due **by the end of the lab day (23:59)**. Submit your code (a commit in your private repository) and upload your predictions to the Kaggle leaderboard. The **description of your Kaggle submission** is the hand-in channel: paste your commit link into it when you upload.

- [ ] notebook committed and pushed to your private repository
- [ ] lab instructor(s) have accepted [collaborator access](../week_01_setup/README.md#create-your-private-repository) and can open the commit link
- [ ] `submission.csv` on the Kaggle leaderboard, at least 80 % (team / display name = your AIS login)
- [ ] commit link pasted into the Kaggle submission description (add your W&B link too if you used W&B)

Points for this task: ask at the lab.

## Resources

- [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [d2l.ai: Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html)
- [The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
