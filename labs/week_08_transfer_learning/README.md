# Week 8 · Transfer learning: ResNet on Oxford-IIIT Pets

Recognise the breed of a cat or dog from a photo: 37 breeds, about 100 training photos per breed. Start from a model pretrained on ImageNet and adapt it to the 37 breeds. Metric: accuracy.

Notebook: [`task_8_pets_transfer_learning.ipynb`](task_8_pets_transfer_learning.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fiit-ba/zneus-2026/blob/main/labs/week_08_transfer_learning/task_8_pets_transfer_learning.ipynb) (in Colab, replace `fiit-ba` in the URL with your GitHub username to open the copy in your fork).

**A GPU is recommended** (a free Colab T4 is enough). On a laptop develop with `SUBSET` set in the configuration cell.

## Data

Kaggle competition: the link is given at the lab. Download and unzip the competition data (about 200 MB) from its *Data* tab into `labs/week_08_transfer_learning/data/kaggle/`, so that the images are at `data/kaggle/train/<id>.jpg` and `data/kaggle/test/<id>.jpg`.

- `train/<id>.jpg` and `train.csv` (`id,breed`): 3 680 training photos with their breed.
- `test/<id>.jpg` and `sample_submission.csv` (`id,breed`): 3 669 test photos, no labels; the sample shows the required format.
- JPEG, RGB, shorter side 256 px, varying aspect ratio. The 37 breed strings are listed in the notebook; your submission must spell them exactly like that.
- Without the Kaggle files the notebook uses the torchvision copy of the Oxford-IIIT Pet dataset (800 MB); that submission will not score on Kaggle.

The notebook builds a table of image paths and labels for you. Everything else, from looking at the data to the final predictions, is your work.

## Task

Fine-tune an ImageNet-pretrained model from `torchvision.models` to the 37 breeds as well as you can, log every run to Weights & Biases (project `zneus-2026`, run names `week08-...`) and submit your predictions for the test photos to Kaggle (`submission.csv`, columns `id,breed`, breed names as strings).

## Hand-in

Due **by the end of the lab day (23:59)**. Every hand-in consists of the code (a commit in your fork) and, for anything that trains a model, a public Weights & Biases link; Kaggle weeks additionally require a submission on the leaderboard.

- [ ] notebook committed and pushed to your fork (commit link)
- [ ] `submission.csv` on the Kaggle leaderboard (team / display name = your AIS login)
- [ ] public W&B link to the project `zneus-2026` with your week 8 runs

Points for this task: ask at the lab.

## Resources

- [torchvision models and pretrained weights](https://pytorch.org/vision/stable/models.html)
- [PyTorch tutorial: Transfer Learning for Computer Vision](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (Parkhi et al., 2012)
