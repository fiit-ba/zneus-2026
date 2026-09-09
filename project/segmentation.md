# Topic 1 · Medical image segmentation

**Goal.** Train a network that takes a medical image and outputs a binary mask: which pixels belong to the lesion, the
polyp, the tumour or the nucleus. Evaluate it with Dice and IoU on a test split that was never used for training or
tuning, compare a U-Net trained from scratch against a U-Net with a pretrained encoder, and show overlays of good and
bad predictions together with an explanation of the failures.


## Why it matters

Segmentation is the workhorse of medical image analysis: measuring a lesion, planning radiotherapy, counting cells,
flagging a polyp during a colonoscopy. It is also the task where the gap between "works in my notebook" and "works in
the clinic" is widest. Datasets are tiny (hundreds to a few thousand images), the positive class is a small fraction of
the pixels, masks are drawn by humans with their own biases, and an image from a different scanner or clinic looks
different. All of these problems appear in this project, at a scale where you can still train on a free GPU.

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
