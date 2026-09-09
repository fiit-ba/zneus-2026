# Topic 3 · Road-scene object detection

**Goal.** Fine-tune an object detector that finds cars, pedestrians and/or traffic signs in road-scene images. Evaluate
it with mAP@0.5 and COCO mAP on a held-out test split, measure its speed on CPU and GPU, map one detector across three
input resolutions on the speed-vs-accuracy plane (a second architecture is strongly suggested if you have the
GPU-hours), and show where the detector fails (small objects, night, occlusion).

## Why it matters

Detection is classification plus localisation, done for every object in the image at once. It is what a
driver-assistance system does thirty times per second under a hard latency budget, so every design decision trades
accuracy against speed. Detection is also the task with the most misunderstood metric: everybody reports mAP, few can
say what it measures. After this project you can, because you will have written it (below).


## Datasets

| Dataset | Images | Classes | Size on disk | Licence | Link | Notes |
|---|---|---|---|---|---|---|
| Penn-Fudan Pedestrians | 170 (345 pedestrians) | 1 | about 50 MB | see page | https://www.cis.upenn.edu/~jshi/ped_html/ | warm-up only; the torchvision tutorial uses it |
| Road Sign Detection (Kaggle) | 877 | 4: traffic light, stop, speed limit, crosswalk | about 230 MB | CC0 | https://www.kaggle.com/datasets/andrewmvd/road-sign-detection | Pascal VOC XML annotations |
| GTSDB (German Traffic Sign Detection Benchmark) | 900 (600 train / 300 test), 1360×800 | 43 sign classes in 4 categories | about 1.6 GB | free for research, see page | https://benchmark.ini.rub.de/gtsdb_dataset.html | signs are 16 to 128 px: a small-object problem |
| KITTI 2D object detection | 7 481 labelled images (the 7 518 test images have no public labels) | 8: car, van, truck, pedestrian, person sitting, cyclist, tram, misc (+ DontCare) | about 12 GB images, labels a few MB | CC BY-NC-SA 3.0 | https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d | `torchvision.datasets.Kitti` downloads and parses it; consecutive frames from drive sequences: a random split leaks; use the standard sequence-disjoint 3 712 / 3 769 train/val split (the file lists are in most KITTI detection repositories) and carve test from the 3 769; download is about 12 GB, plan an hour |
| BDD100K | 100 000 at 1280×720; use the 10k subset or less | 10: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign | 5.3 GB for 100k images, about 1.1 GB for 10k | custom licence, see page | https://bdd-data.berkeley.edu/ · docs: https://doc.bdd100k.com/ | per-image daytime / weather attributes |
| Udacity Self-Driving Car | 9 423 frames (set 1) / 15 000 frames (set 2), 1920×1200 | car, truck, pedestrian (+ traffic lights in set 2) | several GB | MIT | https://github.com/udacity/self-driving-car/tree/master/annotations · 512×512 version: https://public.roboflow.com/object-detection/self-driving-car | consecutive video frames, and a large share of frames had missing labels in the original release (Roboflow's re-annotated version fixed many); if you use it, the data card must count images with zero boxes and check 20 of them by eye |

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
