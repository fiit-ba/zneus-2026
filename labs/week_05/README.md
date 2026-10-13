# Week 5 · MLP for regression: California housing

Predict the median house value of a Californian census block group from 8 numeric features. Metric: RMSE (root mean squared error) in units of 100 000 USD, lower is better.

Notebook: [`task_4_california_housing_mlp.ipynb`](task_4_california_housing_mlp.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiit-ba/zneus-2026/blob/main/labs/week_05/task_4_california_housing_mlp.ipynb)

For your private copy, follow the [Colab setup instructions](../week_01_setup/README.md#google-colab).

## Join the Kaggle competition

Open the [Week 5 invitation link](https://www.kaggle.com/t/a8b12269c59c4476879bd5a70e4ed0c7), sign in to Kaggle (or create an account), then read and accept the competition rules to join. Use your **AIS login** as your team name.

## Data

After joining, download `train.csv`, `test.csv` and `sample_submission.csv` from the competition's *Data* tab into `labs/week_05/data/kaggle/`.

- `train.csv`: 16 512 rows with `id`, the 8 features and the target `MedHouseVal`.
- `test.csv`: 4 128 rows with `id` and the 8 features, no target. Predict these.
- Features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude` (see the notebook for what they mean).
- Without the Kaggle files the notebook uses scikit-learn's copy of the dataset with a local split (that submission will not score on Kaggle).

The public leaderboard uses 50% of the test rows; the remaining 50% determine the final private leaderboard. Work individually, with a limit of 20 submissions per day and one selected final submission.

Use only the supplied training labels; looking up test targets in another copy of the public dataset is not permitted.

The notebook loads the files for you. Everything else, from looking at the data to the final predictions, is your work.

## Task

Train the best MLP you can and submit your predictions for `test.csv` to Kaggle (`submission.csv`, columns `id,MedHouseVal`, one row per test id).

W&B logging is optional for this lab; no W&B link is required. If you do log the runs, add the link to the Kaggle submission description next to your commit link.

## Hand-in

Due **14 October 2026 at 23:59 (Europe/Bratislava, CEST)**. Submit your code (a commit in your private repository) and upload your predictions to the Kaggle leaderboard. The **description of your Kaggle submission** is the hand-in channel: paste your commit link into it when you upload.

- [ ] notebook committed and pushed to your private repository
- [ ] lab instructor(s) have accepted [collaborator access](../week_01_setup/README.md#create-your-private-repository) and can open the commit link
- [ ] `submission.csv` on the Kaggle leaderboard (team / display name = your AIS login)
- [ ] commit link pasted into the Kaggle submission description (add your W&B link too if you used W&B)

Points for this task: ask at the lab.

## Resources

- [PyTorch tutorial: Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), [Build the neural network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html), [Optimizing model parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [scikit-learn: the California housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [Weights & Biases quickstart](https://docs.wandb.ai/quickstart) · [Kaggle: making a submission](https://www.kaggle.com/docs/competitions#making-a-submission)
