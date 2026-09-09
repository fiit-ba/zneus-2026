# Topic 1 · Medical image segmentation

**Goal.** Train a network that takes a medical image and outputs a binary mask: which pixels belong to the lesion, the
polyp, the tumour or the nucleus. Evaluate it with Dice and IoU on a test split that was never used for training or
tuning, compare a U-Net trained from scratch against a U-Net with a pretrained encoder, and show overlays of good and
bad predictions together with an explanation of the failures.

Back to the [project overview](../README.md)

**Extra packages** (see [the README](../README.md#general-requirements-all-topics)):
`uv add segmentation-models-pytorch torchmetrics`; Albumentations is optional.

## Why it matters

Segmentation is the workhorse of medical image analysis: measuring a lesion, planning radiotherapy, counting cells,
flagging a polyp during a colonoscopy. It is also the task where the gap between "works in my notebook" and "works in
the clinic" is widest. Datasets are tiny (hundreds to a few thousand images), the positive class is a small fraction of
the pixels, masks are drawn by humans with their own biases, and an image from a different scanner or clinic looks
different. All of these problems appear in this project, at a scale where you can still train on a free GPU.

## Recap: the task and its metrics

Binary segmentation is per-pixel binary classification. A model maps an image $x \in \mathbb{R}^{3 \times H \times W}$
to a logit map $z \in \mathbb{R}^{1 \times H \times W}$; thresholding $\sigma(z) > 0.5$ gives the predicted mask $P$.
Against the ground-truth mask $G$:

$$
\mathrm{Dice}(P, G) = \frac{2\,|P \cap G|}{|P| + |G|}, \qquad
\mathrm{IoU}(P, G) = \frac{|P \cap G|}{|P \cup G|}, \qquad
\mathrm{IoU} = \frac{\mathrm{Dice}}{2 - \mathrm{Dice}}.
$$

Both are in $[0, 1]$ and IoU is always the lower one. The identity $\mathrm{IoU} = \mathrm{Dice} / (2 - \mathrm{Dice})$
holds per image, not for the means: convert per image, then average, or the columns of your table will not add up.
Medical papers usually report the **mean per-image Dice**
(compute Dice for every test image, then average), which is not the same as Dice over all test pixels pooled together.
Report the per-image mean and standard deviation, and decide (and write down) what a pair of empty masks scores: BUSI
has 133 images with no tumour, and most LGG slices contain no tumour.

**Losses.** `nn.BCEWithLogitsLoss` treats every pixel equally, so with a lesion covering 5 % of the image the network
is happy to predict background everywhere. The soft Dice loss

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon},
\qquad p_i = \sigma(z_i),
$$

optimises the metric directly and is insensitive to the amount of background; BCE + Dice is the standard combination.

**Architecture.** The **U-Net** (an encoder that shrinks the image, a decoder that grows it back, skip connections
between the two at every resolution) is the default for small medical datasets. A U-Net whose encoder is an
ImageNet-pretrained ResNet-34 usually wins on a thousand images. `segmentation_models_pytorch` gives you both in one
line; you may use it if you cite it and can explain what the encoder, the decoder and the skip connections do.

## Datasets

Pick **one primary dataset**. Kvasir-SEG and BUSI are the fastest to start with; LGG has a patient structure that
makes the split question interesting; ISIC 2018 is the largest and most realistic; the Data Science Bowl is the
odd one out (many small objects per image). Add a second dataset only as a stretch goal.

| Dataset | Images | Modality and target | Resolution | Size on disk | Licence | Link |
|---|---|---|---|---|---|---|
| Kvasir-SEG | 1 000 images with masks | colonoscopy, polyp | 332×487 to 1920×1072 | 46 MB | research and education use, see page | https://datasets.simula.no/kvasir-seg/ |
| BUSI (Breast Ultrasound Images) | 780 images with masks: 437 benign, 210 malignant, 133 normal | ultrasound, tumour | about 500×500, grayscale | about 250 MB | CC BY 4.0 | https://scholar.cu.edu.eg/?q=afahmy/pages/dataset · Kaggle mirror: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset |
| LGG MRI segmentation | 3 929 slices from 110 patients (TCGA) | brain MRI (FLAIR), lower-grade glioma | 256×256, 3 channels | about 700 MB | CC BY-NC-SA 4.0 | https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation |
| ISIC 2018 Task 1 | 2 594 train + 100 val + 1 000 test images with masks | dermoscopy, skin lesion | variable, up to several thousand pixels per side | about 10 GB (resize once and cache) | CC BY-NC 4.0 | https://challenge.isic-archive.com/data/#2018 |
| 2018 Data Science Bowl | 670 training images (stage 1) with one mask per nucleus | microscopy, cell nuclei | 256×256 to 1024×1024 | about 80 MB | CC0 | https://www.kaggle.com/c/data-science-bowl-2018 |

Kaggle downloads work from the command line, see [Downloading from Kaggle](../README.md#downloading-from-kaggle).

What to know before you load them:

- **Kvasir-SEG**: masks are JPEG, so values are *near* 0 and 255, not exactly; binarise with a threshold (`> 127`).
  Images contain specular highlights, bubbles and a green position indicator in the corner.
- **BUSI**: masks are separate PNG files; a few images have more than one mask file (`..._mask_1.png`), take the union.
  The 133 normal images have empty masks: decide whether they are part of your task. The set contains duplicate and
  near-duplicate images, some filed under different class labels; a random split leaks them. Run the duplicate check of
  the data card, keep duplicates in the same split (or drop them), and say how many you found.
- **LGG**: one folder per patient, files `TCGA_<site>_<id>_<slice>.tif` and `..._mask.tif`. Roughly two thirds of the
  slices have an empty mask. **Split by patient**, never by slice. The "3 channels" are three MRI sequences
  (pre-contrast, FLAIR, post-contrast), not RGB; ImageNet normalisation of them is arbitrary, say what you do instead.
- **ISIC 2018**: the originals are huge; resize to 256×256 (or 512×512) once, save as PNG or `.npz`, and work from the
  cache. Use the official train / validation / test splits from the same page; do not re-split.
- **DSB 2018**: masks come per nucleus; take their union for semantic segmentation. Instance segmentation (separating
  touching nuclei) is a stretch goal.

## Suggested baseline and steps

0. **Data card** (README requirement 2). 20 random raw image / mask pairs; a histogram of the foreground fraction per
   image (log axis); the number of empty masks; the image size distribution; the duplicate check (md5 of every file,
   plus a perceptual hash or 8×8 thumbnails for near-duplicates); for LGG the number of slices per patient.
1. **Data pipeline.** A `Dataset` that returns `(image, mask)` as float tensors: image `(3, 256, 256)`, normalised;
   mask `(1, 256, 256)` with values in {0, 1}. Resize masks with **nearest-neighbour** interpolation. Use
   `torchvision.transforms.v2` with `tv_tensors.Image` and `tv_tensors.Mask` so flips, rotations and crops are applied
   identically to both (or Albumentations, cited). Plot eight augmented pairs before you train anything.
2. **Splits.** 70 / 15 / 15 train / val / test with a fixed seed, unless the dataset ships official splits (ISIC),
   then use them; grouped by patient for LGG (`sklearn.model_selection.GroupShuffleSplit`). Write the three file lists
   to `splits/{train,val,test}.txt` in your project folder so every experiment uses the same split.
3. **Trivial baselines.** All-background, all-foreground, and the mean training mask thresholded at 0.5. Their Dice
   is the floor every model must beat. Notice that on BUSI "normal" images the all-background predictor cannot be
   beaten: think about what that means for your metric and your empty-mask convention.
4. **U-Net from scratch.** Implement a small U-Net in `torch.nn` (four levels, 32-64-128-256 channels, two 3×3
   convolutions + BatchNorm + ReLU per level, `MaxPool2d` down, `ConvTranspose2d` or upsample + 1×1 conv up, skip
   connections by concatenation). Train with BCE plus your own Dice loss (below), Adam at 1e-3, 30–50 epochs, batch
   8–16 at 256×256. Before the real run do the sanity checks from the README, with the expected numbers for this
   topic: BCE about 0.69 per pixel at init; soft Dice loss about 0.91 for a 5 % lesion ($1 - f / (0.5 + f)$ at
   $p = 0.5$ everywhere); eight images to Dice above 0.99; the watch batch of eight val images logged every epoch.
   Log train loss, val loss and val Dice per epoch to W&B; keep the checkpoint with the best val Dice.
5. **Pretrained encoder.** `smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1)` with ImageNet
   normalisation, the **same budget** as step 4 (same epochs, same augmentation, same split). This is your headline
   comparison: run it with three seeds (README requirement 5).
6. **Improvement iterations** (one change at a time, same budget, choose from): augmentation (flips, 90° rotations,
   scale jitter, brightness/contrast, elastic deformation), loss (BCE vs Dice vs BCE + Dice vs focal), input
   resolution, encoder depth (ResNet-18 / 34 / 50), a learning-rate schedule, threshold tuning on val, post-processing
   (keep the largest connected component with `scipy.ndimage.label`, morphological opening), test-time augmentation
   (average predictions over flips).
7. **Evaluation, once.** Test Dice and IoU (mean ± std over images) for the trivial baseline, the from-scratch U-Net,
   the pretrained U-Net and your best variant, all from the checkpoint chosen on val.
8. **Qualitative results.** A grid `image | ground truth | prediction overlay` for the 4 best and the 4 worst test
   images by Dice, with the Dice value in the title.

Budget: develop on a 100-image subset on the CPU, then run the real experiments on a free Colab or Kaggle GPU.

## From scratch: the Dice loss and the Dice metric

This is the piece you write yourself (README requirement 6). Implement `soft_dice_loss(logits, target, eps)` from the
formula in the recap, in plain `torch`, about ten lines; say whether the sums run over the whole batch or per image and
are then averaged (per image is the usual choice and the one that matches the metric). Implement
`dice_per_image(pred_mask, target_mask, empty_value)`, also about ten lines; the empty-mask convention is an explicit
argument, not a comment. On one batch, assert that your metric agrees with torchmetrics (`DiceScore` in
`torchmetrics.segmentation`, or `torchmetrics.functional.dice`; check which one your torchmetrics version has) to
within 1e-5 on non-empty masks, and show the one case where they differ: the empty / empty pair, where the library
returns its own default and yours returns `empty_value`. Explain the difference in a sentence. Every Dice number in
the report comes from your function.

## Mandatory analyses

- A **results table** with the trivial baseline, the from-scratch U-Net, the pretrained-encoder U-Net and your best
  variant: Dice and IoU as mean ± std over test images, and for the from-scratch vs pretrained comparison mean ± std
  over three seeds; the number of test images; the same split for every row.
- **Learning curves** (train and val loss, val Dice) from W&B for from-scratch vs pretrained. Which one overfits, and
  when? How many epochs did the pretrained encoder need to reach the from-scratch model's final Dice?
- **Overlays** of the best and worst test cases, the worst 20 test images as a gallery (README requirement 8), plus a
  paragraph on what the failures have in common: small lesions, low contrast, hair and ruler marks (ISIC), specular
  reflections and bubbles (Kvasir), shadows and calcifications (BUSI), tiny tumours in otherwise empty slices (LGG),
  touching nuclei (DSB).
- A **per-image Dice histogram** or box plot: is the mean hiding a bimodal distribution (many near-perfect images, a
  few complete misses)?
- **One controlled ablation**: loss, augmentation or encoder, with everything else fixed.
- A written **empty-mask convention** and how many test images it affects.
- **Training cost**: time per epoch and the hardware used.

## Common pitfalls

The pitfalls shared by all topics (evaluating on train, tuning on test, `model.eval()`, budgets, normalisation, seeds)
are in [the README](../README.md#pitfalls-everybody-falls-into); these are specific to segmentation.

- **Patient leakage.** Neighbouring slices of one LGG patient in train and test look nearly identical and inflate Dice
  by a lot. Split by patient.
- **Duplicates across splits (BUSI).** Near-identical images in train and test inflate Dice exactly like patient
  leakage. Hash and check.
- **Resizing masks with bilinear interpolation** produces values between 0 and 1 on the boundary. Use nearest-neighbour
  and binarise (`mask > 0.5`).
- **Masks stored as 0 / 255** (and JPEG masks with values like 3 or 252). Binarise with a threshold; do not divide by
  255 and hope for exact zeros and ones.
- **Class imbalance.** BCE alone converges to all-background on small lesions. Watch the val Dice, not the loss.
- **Empty masks.** Dice is 0 / 0 when both masks are empty. Define it (1 is common), or report positive and empty
  images separately, and say what you did.
- **Augmenting the image but not the mask** (or with different random parameters). Plot augmented pairs.
- **BatchNorm with batch size 1–2** at high resolution is unstable. Use a smaller resolution or a larger batch.
- **Mean Dice only.** A few zero-Dice failures matter clinically. Show the distribution.

## Stretch goals

- **Cross-dataset generalisation**: train on Kvasir-SEG, test on CVC-ClinicDB (612 polyp frames,
  https://polyp.grand-challenge.org/CVCClinicDB/) and explain the drop.
- **Three classes on BUSI**: background / benign / malignant as a multi-class mask; how does the mask quality relate to
  the diagnosis?
- **Instance segmentation on DSB 2018**: watershed on the predicted mask and a distance or boundary map; report the
  competition metric (mean average precision over IoU thresholds).
- **Boundary-aware losses** or deep supervision; compare against nnU-Net's default recipe on your data.
- **Uncertainty**: MC-dropout or a small ensemble. Are the wrong pixels the uncertain ones?
- **Efficiency**: Dice vs inference time for a MobileNet encoder vs a ResNet encoder (this connects to topic 4).

## References

- Read first: Ronneberger, Fischer, Brox (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
  https://arxiv.org/abs/1505.04597
- Read first: Milletari, Navab, Ahmadi (2016). V-Net (introduces the soft Dice loss). https://arxiv.org/abs/1606.04797
- Read first: Reinke et al. (2024). Understanding metric-related pitfalls in image analysis validation.
  https://arxiv.org/abs/2302.01790 (read the Dice / empty-mask sections)
- Isensee et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
  https://arxiv.org/abs/1904.08128 · code: https://github.com/MIC-DKFZ/nnUNet
- Iakubovskii. segmentation_models.pytorch. https://github.com/qubvel-org/segmentation_models.pytorch
- Codella et al. (2019). Skin Lesion Analysis Toward Melanoma Detection 2018 (ISIC 2018).
  https://arxiv.org/abs/1902.03368 · Tschandl, Rosendahl, Kittler (2018). The HAM10000 dataset.
  https://arxiv.org/abs/1803.10417
- Jha et al. (2020). Kvasir-SEG: A Segmented Polyp Dataset. https://arxiv.org/abs/1911.07069
- Al-Dhabyani et al. (2020). Dataset of breast ultrasound images. Data in Brief.
  https://doi.org/10.1016/j.dib.2019.104863
- Buda, Saha, Mazurowski (2019). Association of genomic subtypes of lower-grade gliomas with shape features
  automatically extracted by a deep learning algorithm. https://arxiv.org/abs/1906.03720
- Caicedo et al. (2019). Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl. Nature Methods.
  https://doi.org/10.1038/s41592-019-0612-7
- torchvision transforms v2 (joint image and mask transforms): https://pytorch.org/vision/stable/transforms.html ·
  Albumentations: https://albumentations.ai/ · MONAI (medical imaging toolkit): https://monai.io/
- torchmetrics (Dice, Jaccard index): https://lightning.ai/docs/torchmetrics/stable/
