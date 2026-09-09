# Topic 3 · Road-scene object detection

**Goal.** Fine-tune an object detector that finds cars, pedestrians and/or traffic signs in road-scene images. Evaluate
it with mAP@0.5 and COCO mAP on a held-out test split, measure its speed on CPU and GPU, map one detector across three
input resolutions on the speed-vs-accuracy plane (a second architecture is strongly suggested if you have the
GPU-hours), and show where the detector fails (small objects, night, occlusion).

Back to the [project overview](../README.md)

**Extra packages** (see [the README](../README.md#general-requirements-all-topics)): `uv add torchmetrics pycocotools`
(torchmetrics' mAP needs `pycocotools` or `faster-coco-eval`).

## Why it matters

Detection is classification plus localisation, done for every object in the image at once. It is what a
driver-assistance system does thirty times per second under a hard latency budget, so every design decision trades
accuracy against speed. Detection is also the task with the most misunderstood metric: everybody reports mAP, few can
say what it measures. After this project you can, because you will have written it (below).

## Recap: detectors and mAP

A detector outputs a set of (box, label, score) triples per image. Ground truth is a set of (box, label) pairs. The
overlap of two boxes is their intersection over union,

$$
\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}.
$$

For one class and one IoU threshold $t$: sort predictions by score; walk down the list; a prediction is a **true
positive** if it has $\mathrm{IoU} \ge t$ with a not-yet-matched ground-truth box of the same class, otherwise a
**false positive**; ground-truth boxes never matched are **false negatives**. Sweeping the score threshold gives a
precision-recall curve; the **average precision** (AP) is the area under it (interpolated). PASCAL VOC 2010+ uses the
all-point interpolation; COCO and `torchmetrics` sample precision at 101 recall points; the two differ slightly and
your table must say which one it uses. **mAP** is the mean AP over classes. PASCAL VOC's **mAP@0.5** uses $t = 0.5$; **COCO mAP** averages over $t \in \{0.50, 0.55, \ldots, 0.95\}$ and is
stricter about box placement. COCO also reports AP for small (area < 32²), medium and large (> 96²) objects, which is
the axis road scenes fail on. Overlapping predictions of the same object are removed before evaluation by
non-maximum suppression (NMS).

Detector families in torchvision:

- **Two-stage**: Faster R-CNN. A region proposal network proposes boxes on a feature pyramid (FPN); a head classifies
  and refines them. Accurate, slower.
- **One-stage, anchor-based**: RetinaNet (focal loss to handle the background imbalance), SSDlite (MobileNetV3 backbone,
  very fast, weaker on small objects).
- **One-stage, anchor-free**: FCOS predicts a box at every feature-map location.

`fasterrcnn_mobilenet_v3_large_fpn` is a good first model: it fine-tunes in minutes and runs on a CPU.

## Datasets

| Dataset | Images | Classes | Size on disk | Licence | Link | Notes |
|---|---|---|---|---|---|---|
| Penn-Fudan Pedestrians | 170 (345 pedestrians) | 1 | about 50 MB | see page | https://www.cis.upenn.edu/~jshi/ped_html/ | warm-up only; the torchvision tutorial uses it |
| Road Sign Detection (Kaggle) | 877 | 4: traffic light, stop, speed limit, crosswalk | about 230 MB | CC0 | https://www.kaggle.com/datasets/andrewmvd/road-sign-detection | Pascal VOC XML annotations |
| GTSDB (German Traffic Sign Detection Benchmark) | 900 (600 train / 300 test), 1360×800 | 43 sign classes in 4 categories | about 1.6 GB | free for research, see page | https://benchmark.ini.rub.de/gtsdb_dataset.html | signs are 16 to 128 px: a small-object problem |
| KITTI 2D object detection | 7 481 labelled images (the 7 518 test images have no public labels) | 8: car, van, truck, pedestrian, person sitting, cyclist, tram, misc (+ DontCare) | about 12 GB images, labels a few MB | CC BY-NC-SA 3.0 | https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d | `torchvision.datasets.Kitti` downloads and parses it; consecutive frames from drive sequences: a random split leaks; use the standard sequence-disjoint 3 712 / 3 769 train/val split (the file lists are in most KITTI detection repositories) and carve test from the 3 769; download is about 12 GB, plan an hour |
| BDD100K | 100 000 at 1280×720; use the 10k subset or less | 10: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign | 5.3 GB for 100k images, about 1.1 GB for 10k | custom licence, see page | https://bdd-data.berkeley.edu/ · docs: https://doc.bdd100k.com/ | per-image daytime / weather attributes |
| Udacity Self-Driving Car | 9 423 frames (set 1) / 15 000 frames (set 2), 1920×1200 | car, truck, pedestrian (+ traffic lights in set 2) | several GB | MIT | https://github.com/udacity/self-driving-car/tree/master/annotations · 512×512 version: https://public.roboflow.com/object-detection/self-driving-car | consecutive video frames, and a large share of frames had missing labels in the original release (Roboflow's re-annotated version fixed many); if you use it, the data card must count images with zero boxes and check 20 of them by eye |

Recommendation: warm up on Penn-Fudan with the torchvision tutorial (an hour), then pick **one main dataset**. Road Sign
Detection or GTSDB are the default; KITTI or a 2 000–5 000 image subset of BDD100K only if you have a GPU session you
can rely on and you know what a 12 GB download means on Colab. Do not try to train on all of BDD100K. Kaggle datasets
download from the command line, see [Downloading from Kaggle](../README.md#downloading-from-kaggle).

## Suggested baseline and steps

1. **Learn the API** with the torchvision tutorial. Inputs are a *list* of `(3, H, W)` float tensors in `[0, 1]`
   (images may have different sizes); targets are a list of dicts with `boxes` of shape `(N, 4)` in `xyxy` pixel
   coordinates and `labels` of shape `(N,)`, `int64`, with **0 reserved for background**. In `train()` mode the model
   returns a dict of losses; in `eval()` mode a list of dicts with `boxes`, `labels`, `scores`.
2. **Dataset class** for your data: parse the VOC XML / KITTI text / BDD100K JSON into that format;
   `collate_fn=lambda batch: tuple(zip(*batch))`; transforms with `torchvision.transforms.v2` and
   `tv_tensors.BoundingBoxes` so that flips and crops move the boxes too.
3. **Data card** (README requirement 2). Draw 20 training images with their ground-truth boxes
   (`torchvision.utils.draw_bounding_boxes`) before you train anything. A class histogram. A histogram of box sizes in
   pixels (square root of the area) with the COCO small / medium / large cut-offs marked. Objects per image. The number
   of images with zero boxes (and, for Udacity, whether they really are empty). Duplicate or near-consecutive frames
   (8×8 thumbnails). For GTSDB note how many signs are below 32 px: that number decides your anchors and resolution.
4. **Split** train / val / test by image, or by sequence where frames are consecutive (Udacity: split by time; KITTI:
   the sequence-disjoint 3 712 / 3 769 lists, test carved from the 3 769; BDD100K: one frame per video, so by image is
   fine). Write the file lists to `splits/{train,val,test}.txt` in your project folder.
5. **Baseline.** `fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)`,
   replace `model.roi_heads.box_predictor` with `FastRCNNPredictor(in_features, num_classes + 1)`, SGD with lr 0.005,
   momentum 0.9, weight decay 5e-4 (or AdamW 1e-4), 5–10 epochs. Log the loss components (`loss_classifier`,
   `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`) and the val mAP per epoch to W&B. Before the real run, do
   the README sanity checks with this topic's numbers: at init, say which of the four losses dominates and check it;
   eight images to near-zero box and classification loss in a few hundred steps; a watch batch of eight val images with
   drawn predictions logged every epoch. Note that torchvision detectors apply `box_score_thresh=0.05` and
   `box_detections_per_img=100` internally; for mAP set `box_score_thresh=0.001`, or say that you kept 0.05.
6. **Evaluate** with `torchmetrics.detection.MeanAveragePrecision(iou_type="bbox", class_metrics=True)`: `map_50`,
   `map`, `map_small`, `map_medium`, `map_large`, `map_per_class`. Feed it *all* detections (no score threshold). And
   with your own AP@0.5 (below) for at least one class.
7. **Resolution sweep** (mandatory). The same detector, same split, same epochs at `min_size` 512, 640 and 800;
   mAP@0.5, COCO mAP, AP_small and ms/image for each. This is your speed-vs-accuracy line.
8. **Second detector** (strongly suggested, not mandatory; same data, same epochs): `fasterrcnn_resnet50_fpn_v2`,
   `retinanet_resnet50_fpn_v2`, `fcos_resnet50_fpn`, `ssdlite320_mobilenet_v3_large`. `fasterrcnn_resnet50_fpn_v2` at
   800 px is one to three hours per run on a free T4: checkpoint every epoch. A YOLO-family model via the
   `ultralytics` package (AGPL-3.0, cite it) may be used as a comparison point only, never as the main model, and its
   label conversion and licence go into the report.
9. **Speed.** Time every variant with [How to time a model](../README.md#how-to-time-a-model), on CPU and on GPU.
   torchvision detectors resize internally to `min_size=800` by default; lower it and watch mAP and FPS move in
   opposite directions.
10. **Qualitative results.** Test images with predictions (score ≥ 0.5) next to the ground truth: eight good, eight
    bad, and a list of failure categories.

## From scratch: average precision

Write `ap.py` with `average_precision(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)` for one class over the
whole test split: sort the predictions by score; walk down the list and greedily match each one to the not-yet-matched
ground-truth box of highest IoU (the IoU is a few lines and yours too); cumulative true and false positives; the
precision and recall arrays; the area under the precision envelope (all-point interpolation). About forty lines.
Assert it against `MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)` on identical inputs and state the
difference: 101-point versus all-point interpolation makes it small but not zero on a few hundred boxes. Say which of
the two numbers your table uses. Then plot your precision-recall curve for that class: that is the mandatory PR curve.

## Mandatory analyses

- **Table**: one row per (detector, resolution): detector | backbone | resolution | parameters | mAP@0.5 | COCO mAP |
  AP_small | ms/image CPU | ms/image GPU, all on the same test split with the number of test images stated, and
  mean ± std over three seeds for the headline comparison (baseline resolution vs best).
- **Speed-vs-accuracy scatter plot**: one point per (detector, resolution).
- **Per-class AP** and a discussion of rare classes. Say what you did with classes that have fewer than about 20
  instances (merged into categories? dropped?).
- **AP by object size** (small / medium / large), or a plot of recall against ground-truth box size.
- **A precision-recall curve** for at least one class, from your `ap.py`, plus precision and recall at a chosen score
  threshold with a justification of the threshold (a deployment decision, not a metric decision).
- **Qualitative results and failure modes**. For BDD100K additionally mAP by daytime vs night and by weather, using the
  per-image attributes in the labels.
- **Training curves** in W&B: all loss components and val mAP per epoch.

## Common pitfalls

The pitfalls shared by all topics are in [the README](../README.md#pitfalls-everybody-falls-into); these are specific
to detection.

- **Label 0 is background.** `num_classes` includes it; an off-by-one gives silently wrong class names.
- **Box formats.** torchvision: `xyxy` absolute pixels. COCO: `xywh`. YOLO: normalised `cxcywh`. KITTI: left, top,
  right, bottom. BDD100K: `x1, y1, x2, y2`. Convert and *plot* to verify.
- **Degenerate boxes** (width or height ≤ 0 after cropping and rounding) crash training with a cryptic error. Filter
  them. Images without objects need `boxes` of shape `(0, 4)`, not an empty list.
- **`torch.stack` on images of different sizes.** Pass a list; the model handles batching.
- **Sequence leakage** (KITTI, Udacity): frames a tenth of a second apart in train and test. Split by sequence or time.
- **Thresholding scores before computing mAP.** mAP integrates over the score axis; feed it every detection (torchmetrics
  handles the cap). Score thresholds are for pictures and deployment metrics.
- **Tiny test sets** make mAP unstable; use cross-validation on Penn-Fudan.
- **Anchor sizes** in the pretrained models are tuned for COCO objects; GTSDB signs at 16–32 px fall below the smallest
  default anchor of several models. Increase `min_size`, change the `AnchorGenerator` sizes, or crop tiles. Check
  `map_small`.
- **GTSDB's 43 classes** with a handful of examples each: merge them into the four categories (prohibitory, mandatory,
  danger, other) for detection, or detect "sign" and classify the crop with a separate classifier.
- **KITTI `DontCare` regions**: exclude them from the targets, and ideally ignore predictions inside them when
  evaluating; at least say what you did.
- **Slow CPU training**: freeze more backbone layers (`trainable_backbone_layers`), use the MobileNetV3 backbone, lower
  `min_size`, train on a subset, and do the real runs on a GPU.
- **Augmentations that break boxes** (rotation, heavy crops). Horizontal flip, scale jitter and photometric changes are
  safe; check the augmented boxes visually.
- **Comparing detectors at different input resolutions** without saying so. Resolution is the strongest knob you have.
- **Class imbalance** (cars outnumber everything; trains are rare in BDD100K). Per-class AP shows it; mAP hides it.

## Stretch goals

- **YOLO comparison** at equal FPS (as a comparison point only, see step 8) and a discussion of the licence.
- **Cross-dataset evaluation**: train on KITTI, test on BDD100K (map the class names) and explain the drop.
- **Night vs day**: augment with brightness and gamma to simulate night; does night mAP improve?
- **Deploy**: export the best detector to ONNX Runtime, measure CPU FPS, try int8 (this connects to topic 4).
- **Tracking**: link detections across Udacity's consecutive frames with IoU matching and count vehicles in a clip.
- **Test-time augmentation** (horizontal flip) and its cost in FPS.
- **Error diagnosis** with TIDE: how much mAP is lost to classification, localisation, duplicates, background, misses?

## References

- Read first: Ren, He, Girshick, Sun (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
  Networks. https://arxiv.org/abs/1506.01497
- Lin et al. (2017). Feature Pyramid Networks for Object Detection. https://arxiv.org/abs/1612.03144
- Lin et al. (2017). Focal Loss for Dense Object Detection (RetinaNet). https://arxiv.org/abs/1708.02002
- Tian et al. (2019). FCOS: Fully Convolutional One-Stage Object Detection. https://arxiv.org/abs/1904.01355
- Liu et al. (2016). SSD: Single Shot MultiBox Detector. https://arxiv.org/abs/1512.02325 · Howard et al. (2019).
  Searching for MobileNetV3. https://arxiv.org/abs/1905.02244
- Lin et al. (2014). Microsoft COCO: Common Objects in Context. https://arxiv.org/abs/1405.0312 · Read first: COCO
  detection evaluation: https://cocodataset.org/#detection-eval
- Everingham et al. The PASCAL Visual Object Classes Challenge. http://host.robots.ox.ac.uk/pascal/VOC/
- Read first: Bolya et al. (2020). TIDE: A General Toolbox for Identifying Object Detection Errors.
  https://arxiv.org/abs/2008.08115
- Houben et al. (2013). Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark.
  https://benchmark.ini.rub.de/gtsdb_dataset.html
- Geiger, Lenz, Urtasun (2012). Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite.
  https://www.cvlibs.net/datasets/kitti/
- Yu et al. (2020). BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning.
  https://arxiv.org/abs/1805.04687
- torchvision object detection fine-tuning tutorial (Penn-Fudan):
  https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- torchvision detection models and weights: https://pytorch.org/vision/stable/models.html#object-detection
- torchmetrics `MeanAveragePrecision`:
  https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
- Ultralytics YOLO documentation (AGPL-3.0): https://docs.ultralytics.com/
