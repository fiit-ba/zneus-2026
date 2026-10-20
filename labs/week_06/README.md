# Week 6 · MLP for classification: MNIST

Classify handwritten digits (0–9) from 28 × 28 grayscale images. Train a fully connected neural network in PyTorch. Metric: accuracy, higher is better.

Notebook: [`task_5_mnist_mlp.ipynb`](task_5_mnist_mlp.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiit-ba/zneus-2026/blob/main/labs/week_06/task_5_mnist_mlp.ipynb)

For your private copy, follow the [Colab setup instructions](../week_01_setup/README.md#google-colab).

## Join the Kaggle competition

Open the [Week 6 invitation link](https://www.kaggle.com/t/e3b24b434743440c88a614faad67367e), sign in to Kaggle, then read and accept the competition rules to join. Use your **AIS login** as your team name.

## Data

Download and extract the competition files into `labs/week_06/data/kaggle/`.

- `train.csv`: 60 000 rows with `id`, `label` and 784 pixel columns.
- `test.csv`: 10 000 rows with `id` and 784 pixel columns; predict the missing labels.
- `sample_submission.csv`: the required format, `id,label`, with dummy predictions.
- `pixel0` to `pixel783`: grayscale intensities from 0 to 255, ordered row by row from the top left of each image. Reshape a row to `(28, 28)` to display it.

The competition uses its own shuffled, stratified split of MNIST with new IDs. The public leaderboard uses 5 000 test images; the remaining 5 000 determine the final private leaderboard. Work individually, with 20 submissions per day and one selected final submission.

Use only the supplied training labels. Looking up test labels in another copy of MNIST, matching test images to labelled copies, external training data and pretrained models are not permitted.

Without the Kaggle files, the notebook can use torchvision's MNIST for local practice. That mode writes `submission_local.csv`, which is **not a Kaggle submission**.

MNIST is credited to Yann LeCun and Corinna Cortes and distributed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), as documented by [Keras](https://keras.io/api/datasets/mnist/). For this lab, the original images were pooled, split again with class stratification, assigned new IDs and exported as CSV; pixel values are unchanged.

## Task

Train the best **MLP** you can, using fully connected layers (no convolutions). Reserve a validation set from the labelled training rows, scale the pixels, train your model and inspect its mistakes. Keep the test images for final predictions.

Submit `submission.csv` with columns `id,label`, one integer digit from 0 to 9 for every test ID. The notebook provides data loading and submission checks; model design, training and evaluation are your work.

W&B logging is optional; no W&B link is required. If you use it, add the run or project link beside your commit link.

## Hand-in

Due **21 October 2026 at 23:59 (Europe/Bratislava, CEST)**. The **description of your Kaggle submission** is the hand-in channel: paste the link to the commit containing your notebook there.

- [ ] notebook runs top to bottom, is committed and pushed to your private repository
- [ ] lab instructor(s) have accepted [collaborator access](../week_01_setup/README.md#create-your-private-repository) and can open the commit link
- [ ] `submission.csv` is on the Kaggle leaderboard under your AIS login
- [ ] commit link is in the Kaggle submission description (add your W&B link if used)

Points for this task: ask at the lab.

## Resources

- [PyTorch: Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html), [Build the neural network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html), [Optimizing model parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) · [torchvision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
