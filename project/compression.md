# Topic 4 · Make ResNet-50 small and fast

**Goal.** Take a ResNet-50 fine-tuned on an image classification dataset of your choice and make inference cheaper:
faster on a CPU and/or smaller on disk, while losing as little accuracy as possible. A suggested target is **at least 3×
lower latency or at least 4× smaller size at no more than 2 percentage points of accuracy drop**; the exact target is
yours to set and to argue for. The deliverable is a **Pareto plot of accuracy against latency** (and size) across every
variant you built, with an explanation of why each variant sits where it sits.

**Rules.** ResNet-50 must be in the pipeline: as the model you prune, quantise or factorise, or as the **teacher** that
distils into a smaller student. Picking a dataset and training a small network from scratch without ResNet-50 does not
count.

## Why it matters

Accuracy is one axis of a model; cost is the other. A model that runs on a server at 40 ms per image costs money at
scale, and a model that has to run on a phone, a camera or a factory PLC without a GPU has a hard latency and memory
budget. Compression is the toolbox for the second axis: pruning, quantisation, distillation, low-rank factorisation,
and compilers. The hard part is **measuring** correctly and understanding *why* a variant is faster or not. Many
published speed-ups evaporate under a fair protocol; yours should not.

## Datasets

Any image classification dataset on which a fine-tuned ResNet-50 reaches a sensible accuracy. Small and fast to
iterate on is a feature here: the project is about the model, not the data.

| Dataset | Images | Classes | Size on disk | Licence | Link | Notes |
|---|---|---|---|---|---|---|
| Oxford-IIIT Pets | 7 349 (3 680 trainval / 3 669 test) | 37 breeds | about 800 MB | CC BY-SA 4.0 | https://www.robots.ox.ac.uk/~vgg/data/pets/ | you fine-tuned a ResNet on it in week 8; the week 8 Kaggle test split has no public labels, so carve val (500 images) and test (1 000) from the 3 680 trainval images with a fixed seed, or use `torchvision.datasets.OxfordIIITPet(split="test")`, and never score on the Kaggle test. You may reuse your week 8 checkpoint if it was trained on images that are not in your new val/test |
| Imagenette | 9 469 train / 3 925 val | 10 easy ImageNet classes | 99 MB (160 px) / 341 MB (320 px) / 1.5 GB (full) | ImageNet terms, see page | https://github.com/fastai/imagenette | fastest to iterate on |
| CIFAR-100 | 50 000 train / 10 000 test, 32×32 | 100 | 160 MB | see page | https://www.cs.toronto.edu/~kriz/cifar.html | tiny images: upsample to at least 128 px for an ImageNet-pretrained ResNet-50 or accept lower accuracy; `torchvision.datasets.CIFAR100` |
| Food-101 | 101 000 (750 train + 250 test per class) | 101 | about 5 GB | see page | https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ | take 10–20 classes; `torchvision.datasets.Food101` |

## The measurement protocol

Report all of the following for **every** variant, measured the same way, in the same session, on the same machine.

| Quantity | How |
|---|---|
| Parameters | `sum(p.numel() for p in model.parameters())`; for pruned models also the non-zero count (`torch.count_nonzero`) |
| Size on disk | `torch.save(model.state_dict(), f)` then `os.path.getsize(f)` in MB; for unstructured pruning also the gzip-compressed size or a sparse-tensor export, and say which one you report |
| FLOPs (optional) | `torch.utils.flop_counter.FlopCounterMode` around one forward pass |
| CPU latency | exactly as in [How to time a model](../README.md#how-to-time-a-model): batch 1, fixed input, warm-up, median and p90, fixed thread count, CPU model reported; additionally report the quantisation backend (`torch.backends.quantized.engine`) |
| Throughput (optional) | images per second at batch 32 or 64, same warm-up and repetition rules |
| Accuracy | top-1 on the same untouched test split for every variant; the delta in percentage points against the fp32 ResNet-50. The sensitivity study (step 5) chooses its sparsity or temperature on the carved val split; the test split is scored once per variant |
| Pareto plot | x = latency in ms (log scale), y = accuracy; marker size or label = size on disk; the fp32 ResNet-50 highlighted; the Pareto-optimal points connected |


## References

- He, Zhang, Ren, Sun (2015). Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
- Han, Pool, Tran, Dally (2015). Learning both Weights and Connections for Efficient Neural Networks.
  https://arxiv.org/abs/1506.02626 · Han, Mao, Dally (2016). Deep Compression. https://arxiv.org/abs/1510.00149
- Li et al. (2017). Pruning Filters for Efficient ConvNets. https://arxiv.org/abs/1608.08710 · Liu et al. (2017).
  Learning Efficient Convolutional Networks through Network Slimming. https://arxiv.org/abs/1708.06519
- Frankle, Carbin (2019). The Lottery Ticket Hypothesis. https://arxiv.org/abs/1803.03635
- Read first: Blalock et al. (2020). What is the State of Neural Network Pruning? https://arxiv.org/abs/2003.03033
  (before you measure anything)
- Fang et al. (2023). DepGraph: Towards Any Structural Pruning. https://arxiv.org/abs/2301.12900 · Torch-Pruning:
  https://github.com/VainF/Torch-Pruning
- Jacob et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.
  https://arxiv.org/abs/1712.05877
- Krishnamoorthi (2018). Quantizing deep convolutional networks for efficient inference: A whitepaper.
  https://arxiv.org/abs/1806.08342 · Read first: Nagel et al. (2021). A White Paper on Neural Network Quantization.
  https://arxiv.org/abs/2106.08295
- Read first: Hinton, Vinyals, Dean (2015). Distilling the Knowledge in a Neural Network.
  https://arxiv.org/abs/1503.02531
- Beyer et al. (2022). Knowledge distillation: A good teacher is patient and consistent. https://arxiv.org/abs/2106.05237
- Denton et al. (2014). Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation.
  https://arxiv.org/abs/1404.0736 · Jaderberg, Vedaldi, Zisserman (2014). Speeding up Convolutional Neural Networks
  with Low Rank Expansions. https://arxiv.org/abs/1405.3866
- Howard et al. (2019). Searching for MobileNetV3. https://arxiv.org/abs/1905.02244
- PyTorch pruning tutorial: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html · `torch.nn.utils.prune`:
  https://pytorch.org/docs/stable/nn.html#utilities
- PyTorch quantization: https://pytorch.org/docs/stable/quantization.html · torchao: https://github.com/pytorch/ao ·
  torchvision quantizable models: https://pytorch.org/vision/stable/models.html#quantized-models
- PyTorch knowledge distillation tutorial: https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
- `torch.compile`: https://pytorch.org/docs/stable/torch.compiler.html · `torch.utils.benchmark`:
  https://pytorch.org/docs/stable/benchmark_utils.html
- ONNX export: https://pytorch.org/docs/stable/onnx.html · ONNX Runtime: https://onnxruntime.ai/
