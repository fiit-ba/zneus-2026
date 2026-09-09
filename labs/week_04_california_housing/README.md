# Week 4 · MLP for regression: California housing

Predict the median house value of a Californian census block group from 8 numeric features. Metric: RMSE (root mean squared error) in units of 100 000 USD, lower is better.

Notebook: [`task_4_california_housing_mlp.ipynb`](task_4_california_housing_mlp.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiit-ba/zneus-2026/blob/main/labs/week_04_california_housing/task_4_california_housing_mlp.ipynb) (in Colab, replace `fiit-ba` in the URL with your GitHub username to open the copy in your fork).

## Data

Kaggle competition: the link is given at the lab. Download `train.csv`, `test.csv` and `sample_submission.csv` from its *Data* tab into `labs/week_04_california_housing/data/kaggle/`.

- `train.csv`: 16 512 rows with `id`, the 8 features and the target `MedHouseVal`.
- `test.csv`: 4 128 rows with `id` and the 8 features, no target. Predict these.
- Features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude` (see the notebook for what they mean).
- Without the Kaggle files the notebook uses scikit-learn's copy of the dataset with a local split (that submission will not score on Kaggle).

The notebook loads the files for you. Everything else, from looking at the data to the final predictions, is your work.

## Task

Train the best MLP you can, log every run to Weights & Biases (project `zneus-2026`, run names `week04-...`) and submit your predictions for `test.csv` to Kaggle (`submission.csv`, columns `id,MedHouseVal`, one row per test id).

## Hand-in

Due **by the end of the lab**. Every hand-in consists of the code (a commit in your fork) and, for anything that trains a model, a public Weights & Biases link; Kaggle weeks additionally require a submission on the leaderboard.

- [ ] notebook committed and pushed to your fork (commit link)
- [ ] `submission.csv` on the Kaggle leaderboard (team / display name = your AIS login)
- [ ] public W&B link to the project `zneus-2026` with your week 4 runs

Points for this task: ask at the lab.

## Resources

- [PyTorch tutorial: Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), [Build the neural network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html), [Optimizing model parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [scikit-learn: the California housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [Weights & Biases quickstart](https://docs.wandb.ai/quickstart) · [Kaggle: making a submission](https://www.kaggle.com/docs/competitions#making-a-submission)
