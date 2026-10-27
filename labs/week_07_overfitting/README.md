# Week 7 · Something is wrong with this model

Classify 28 x 28 grayscale images of clothing into 10 categories, with a catch: you get only 2 000 training images (200 per class), while the test set has 10 000. A colleague trained an MLP on all 2 000 images: training accuracy 100 %, leaderboard score about 0.80. Beat it. Metric: accuracy.

Notebook: [`task_6_fashion_mnist_2k.ipynb`](task_6_fashion_mnist_2k.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiit-ba/zneus-2026/blob/main/labs/week_07_overfitting/task_6_fashion_mnist_2k.ipynb)

For your private copy, follow the [Colab setup instructions](../week_01_setup/README.md#google-colab).

## Join the Kaggle competition

Open the [Week 7 invitation link](https://www.kaggle.com/t/0ebe475bb0374cc6bbccdf9d0ee0516e), sign in to Kaggle, then read and accept the competition rules to join. Use your **AIS login** as your team name.

## Data

After joining the [Kaggle competition](https://www.kaggle.com/t/0ebe475bb0374cc6bbccdf9d0ee0516e), download `train.npz` and `test.npz` from its *Data* tab into `labs/week_07_overfitting/data/kaggle/`.

- `train.npz`: `x`, a `uint8` array `(2000, 28, 28)`, and `y`, the labels 0..9 (200 per class). **Train only on these 2 000 images.**
- `test.npz`: `x`, 10 000 test images `(10000, 28, 28)`, no labels, shuffled; row `i` is the id `i` of your submission.
- Labels: 0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot.
- Without the Kaggle files the notebook rebuilds the same 2 000 images from the torchvision copy of Fashion-MNIST (30 MB); that submission will not score on Kaggle.

The notebook loads the files for you. Everything else, from looking at the data to the final predictions, is your work.

## Task

Train the best classifier you can on the 2 000 images and submit your predictions for `test.npz` to Kaggle (`submission.csv`, columns `id,label`, 10 000 rows). Beat 0.80 on the leaderboard.

W&B logging is optional for this lab; no W&B link is required. If you do log the runs, add the link to the Kaggle submission description next to your commit link.

## Hand-in

Due **by the end of the lab day (23:59)**. Submit your code (a commit in your private repository) and upload your predictions to the Kaggle leaderboard. The **description of your Kaggle submission** is the hand-in channel: paste your commit link into it when you upload.

- [ ] notebook committed and pushed to your private repository
- [ ] lab instructor(s) have accepted [collaborator access](../week_01_setup/README.md#create-your-private-repository) and can open the commit link
- [ ] `submission.csv` on the Kaggle leaderboard above 0.80 (team / display name = your AIS login)
- [ ] commit link pasted into the Kaggle submission description (add your W&B link too if you used W&B)

Points for this task: ask at the lab.

## Resources

- [d2l.ai: Generalization](https://d2l.ai/chapter_linear-regression/generalization.html)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research)
- [W&B: compare runs](https://docs.wandb.ai/guides/app/features/panels/run-comparer/)
