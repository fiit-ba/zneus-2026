# Topic 2 · Sign-language classification that generalises

**Goal.** Train a classifier that recognises fingerspelled letters of the American manual alphabet from a single image,
and **prove that it works on data from a different source than the training set**: another dataset, or photos of at
least five different people taken by you. Report in-distribution and out-of-distribution accuracy, explain the gap,
and show which remedies close it and by how much.

## Why it matters

Fingerspelling classifiers reach 99 % on the Kaggle datasets in an afternoon and fail the moment a different person
signs in front of a different wall. The public datasets contain one signer, one room and thousands of near-duplicate
video frames, so a random train/test split measures memorisation, and a better model does not change that. Detecting
this situation, measuring it honestly and fixing it is a skill that transfers to every applied machine-learning project.
It also has a human side: assistive tools for Deaf and hard-of-hearing users are only useful if they work for everyone,
not for the one person who recorded the dataset.

## Datasets

| Dataset | Images | Classes | Format | Size on disk | Licence | Link |
|---|---|---|---|---|---|---|
| Sign Language MNIST | 27 455 train + 7 172 test | 24 letters (no J and Z: they involve motion) | 28×28 grayscale, CSV | about 100 MB | CC0 | https://www.kaggle.com/datasets/datamunge/sign-language-mnist |
| ASL Alphabet | 87 000 train (3 000 per class) + 29 official test images | 29: A–Z, space, delete, nothing | 200×200 RGB JPEG | about 1 GB | GPL-2.0 | https://www.kaggle.com/datasets/grassknoted/asl-alphabet |
| ASL Alphabet Test | about 870 (30 per class; a different signer, varied backgrounds) | 29, same as above | RGB JPEG | about 30 MB | see page | https://www.kaggle.com/datasets/danrasband/asl-alphabet-test |
| ASL Fingerspelling A (Pugeault & Bowden) | more than 65 000 | 24 letters | RGB + depth, 5 signers, hands already cropped | several GB (download only the RGB part you need) | research use, see page | https://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset |
| Synthetic ASL Alphabet | 27 000 rendered images | 27: A–Z, blank | RGB, varied synthetic hands and backgrounds | see page | see page | https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet |
| Your own photos | at least 5 people × 24 letters × a few shots | 24 letters | phone camera | – | yours; ask for consent | – |


### The hard requirement: an OOD evaluation

You must evaluate on data from a different source than you trained on. Two ways to satisfy it:

1. **Another dataset.** For example: train on ASL Alphabet, test on Pugeault & Bowden (crop your training images to the
   hand first, or accept the unfairness and discuss it) and on ASL Alphabet Test; or train on Sign Language MNIST, test on
   ASL Alphabet converted to 28×28 grayscale with the same preprocessing function.
2. **Your own photos.** At least **five different people** (not only you), several backgrounds, several lighting
   conditions, left and right hands welcome. A suggested minimum that fits in an afternoon: 24 letters × 5 people × 3
   shots, about 360 photos. Ask every person for consent, frame the hand so that faces are not in the photo (or blur
   them), and keep the photos out of the public fork if the people prefer that; the evaluation code and the resulting
   numbers must be in the fork either way. Label 50 of the photos yourself, blind (file names hidden), before you run
   any model; if a person gets 60 %, the photos are the problem, not the model, and that number goes into the data
   card.


## References

- Pugeault, Bowden (2011). Spelling It Out: Real-Time ASL Fingerspelling Recognition. ICCV Workshops. Dataset and paper:
  https://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset
- Read first: Torralba, Efros (2011). Unbiased Look at Dataset Bias. CVPR.
  https://people.csail.mit.edu/torralba/publications/datasets_cvpr11.pdf
- Read first: Geirhos et al. (2020). Shortcut Learning in Deep Neural Networks. https://arxiv.org/abs/2004.07780
- Recht et al. (2019). Do ImageNet Classifiers Generalize to ImageNet? https://arxiv.org/abs/1902.10811
- Koh et al. (2021). WILDS: A Benchmark of in-the-Wild Distribution Shifts. https://arxiv.org/abs/2012.07421
- Zhang et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. https://arxiv.org/abs/2006.10214 ·
  Hand Landmarker guide: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- Read first: Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
  https://arxiv.org/abs/1610.02391 · implementation: https://github.com/jacobgil/pytorch-grad-cam
- Müller, Hutter (2021). TrivialAugment. https://arxiv.org/abs/2103.10158 · Cubuk et al. (2019). RandAugment.
  https://arxiv.org/abs/1909.13719
- torchvision transforms v2: https://pytorch.org/vision/stable/transforms.html
- Binomial confidence intervals (Wilson score interval):
  https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
