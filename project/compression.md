# Topic 4 · Make ResNet-50 small and fast

**Goal.** Take a ResNet-50 fine-tuned on an image classification dataset of your choice and make inference cheaper:
faster on a CPU and/or smaller on disk, while losing as little accuracy as possible. A suggested target is **at least 3×
lower latency or at least 4× smaller size at no more than 2 percentage points of accuracy drop**; the exact target is
yours to set and to argue for. The deliverable is a **Pareto plot of accuracy against latency** (and size) across every
variant you built, with an explanation of why each variant sits where it sits.

**Rules.** ResNet-50 must be in the pipeline: as the model you prune, quantise or factorise, or as the **teacher** that
distils into a smaller student. Picking a dataset and training a small network from scratch without ResNet-50 does not
count.

Back to the [project overview](../README.md)

**Extra packages** (see [the README](../README.md#general-requirements-all-topics)): `uv add torch-pruning`;
`onnx onnxruntime` for the ONNX stretch goal; `torchao` if the eager quantisation path is gone from your PyTorch (see
below).

## Why it matters

Accuracy is one axis of a model; cost is the other. A model that runs on a server at 40 ms per image costs money at
scale, and a model that has to run on a phone, a camera or a factory PLC without a GPU has a hard latency and memory
budget. Compression is the toolbox for the second axis: pruning, quantisation, distillation, low-rank factorisation,
and compilers. The hard part is **measuring** correctly and understanding *why* a variant is faster or not. Many
published speed-ups evaporate under a fair protocol; yours should not.

## Recap: where the cost is, and the four families

Reference numbers at 224×224: ResNet-50 has 25.6 M parameters, about 100 MB in fp32 (25.6 M × 4 bytes) and about 4.1 GFLOPs
(multiply-adds) per image; ResNet-18 has 11.7 M, 45 MB, 1.8 GFLOPs; MobileNetV3-Large has 5.5 M, 22 MB, 0.22 GFLOPs.
CPU latency is dominated by the convolutions; size is dominated by the weights (4 bytes each in fp32).

1. **Pruning.** *Unstructured* pruning zeroes individual weights (magnitude pruning: remove the smallest);
   sparsity $s$ = fraction of zeros. Dense kernels do **not** get faster, and `state_dict` does **not** get smaller
   unless you store the tensors in a sparse or compressed format. *Structured* (channel / filter) pruning removes whole
   output channels and the matching input channels of the next layer, giving smaller dense tensors and real speed-ups.
   Residual connections couple layers (the channels added together must match), which is why libraries such as
   Torch-Pruning build a dependency graph. Always fine-tune after pruning; iterate (prune 20 %, fine-tune, repeat).
2. **Quantisation.** fp32 to int8: 4× smaller weights and faster integer kernels on CPUs. A tensor is mapped as
   $x_q = \mathrm{round}(x / s) + z$ with a scale $s$ and zero-point $z$ per tensor or per channel.
   *Dynamic* quantisation quantises weights ahead of time and activations on the fly, but in eager mode it covers
   `nn.Linear` (and RNNs) only, so it does almost nothing for ResNet-50, whose cost is in convolutions: a useful
   negative result. *Static post-training quantisation (PTQ)* fuses conv + BN + ReLU, inserts observers, calibrates on
   a hundred or so training batches and converts the whole network. *Quantisation-aware training (QAT)* simulates
   quantisation during fine-tuning and recovers most of the lost accuracy. CPU backends: `fbgemm` / `x86` on Intel and
   AMD, `qnnpack` on ARM (Apple Silicon, Raspberry Pi). `torch.ao.quantization` eager mode and
   `torchvision.models.quantization.resnet50` have been on a deprecation path towards `torchao` for several PyTorch
   releases. Verified on 2026-09-09 with the pinned environment (torch 2.14.0, torchvision 0.29.0, Python 3.12) on
   Apple Silicon with the `qnnpack` engine: fuse, prepare, calibrate, convert and the int8 forward pass all work, but
   every call prints a `DeprecationWarning`, so expect the path to disappear in a later release; `x86` / `fbgemm` on
   Intel and AMD was not tested. Run the same ten-line smoke test yourself on your machine before you plan around it
   (fuse, prepare, calibrate on one batch, convert, one forward pass); if it fails, use the `torchao` PTQ path and cite
   it.
3. **Knowledge distillation.** A small student (ResNet-18, MobileNetV3, a narrow ResNet) is trained on the teacher's
   soft targets:

   $$
   \mathcal{L} = \alpha\, T^2\, \mathrm{KL}\!\left(\mathrm{softmax}(z_t / T)\,\middle\|\,\mathrm{softmax}(z_s / T)\right)
   + (1 - \alpha)\, \mathrm{CE}(y, z_s),
   $$

   with temperature $T \in [2, 8]$ and $\alpha \in [0.5, 0.9]$. The $T^2$ compensates for the gradient scaling by
   $1/T^2$. The **control experiment** is the same student trained on the hard labels with the same budget; the
   difference is what distillation bought you.
4. **Low-rank factorisation.** A weight matrix $W \in \mathbb{R}^{m \times n}$ is replaced by $U V$ with rank $r$,
   $(m + n) r$ parameters instead of $mn$. A $k \times k$ convolution from $C_{\text{in}}$ to $C_{\text{out}}$
   channels becomes a $k \times k$ convolution to $r$ channels followed by a $1 \times 1$ convolution to
   $C_{\text{out}}$; initialise from the SVD of the reshaped kernel and fine-tune. Works best on the widest layers.

**Deployment options** (orthogonal, apply to any variant): `model.to(memory_format=torch.channels_last)`,
`torch.compile(model)`, ONNX Runtime (graph fusions and its own int8 path), and lowering the input resolution
(160 px instead of 224 px cuts the FLOPs roughly in half). Batch size changes throughput, not latency.

## Datasets

Any image classification dataset on which a fine-tuned ResNet-50 reaches a sensible accuracy. Small and fast to
iterate on is a feature here: the project is about the model, not the data.

| Dataset | Images | Classes | Size on disk | Licence | Link | Notes |
|---|---|---|---|---|---|---|
| Oxford-IIIT Pets | 7 349 (3 680 trainval / 3 669 test) | 37 breeds | about 800 MB | CC BY-SA 4.0 | https://www.robots.ox.ac.uk/~vgg/data/pets/ | you fine-tuned a ResNet on it in week 8; the week 8 Kaggle test split has no public labels, so carve val (500 images) and test (1 000) from the 3 680 trainval images with a fixed seed, or use `torchvision.datasets.OxfordIIITPet(split="test")`, and never score on the Kaggle test. You may reuse your week 8 checkpoint if it was trained on images that are not in your new val/test |
| Imagenette | 9 469 train / 3 925 val | 10 easy ImageNet classes | 99 MB (160 px) / 341 MB (320 px) / 1.5 GB (full) | ImageNet terms, see page | https://github.com/fastai/imagenette | fastest to iterate on |
| CIFAR-100 | 50 000 train / 10 000 test, 32×32 | 100 | 160 MB | see page | https://www.cs.toronto.edu/~kriz/cifar.html | tiny images: upsample to at least 128 px for an ImageNet-pretrained ResNet-50 or accept lower accuracy; `torchvision.datasets.CIFAR100` |
| Food-101 | 101 000 (750 train + 250 test per class) | 101 | about 5 GB | see page | https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ | take 10–20 classes; `torchvision.datasets.Food101` |

## The measurement protocol (half of the project)

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

Put the protocol into a script `bench.py` that takes a checkpoint and prints one CSV row, and run every variant through
it. Also record the hardware you fine-tuned on and how long each fine-tuning took.

## Suggested baseline and steps

1. **Baseline.** Fine-tune `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)` on your dataset: replace the fc layer,
   train only the head for 2–3 epochs at lr 1e-3 with the backbone frozen (a linear probe), then unfreeze and train
   5–10 epochs with the backbone at lr 1e-4 and the head at 1e-3, cosine schedule, ImageNet normalisation, 224 px;
   record per-class accuracy. Or reuse your week 8 checkpoint (see the dataset table). Fix the test split. Before the
   real run do the README sanity checks with this topic's numbers: cross-entropy at init about ln 37 ≈ 3.61 on Pets,
   eight images to 100 %, and one extra: run `bench.py` on the fp32 model twice in a row and check that the medians
   agree within 5 %, otherwise your timing is not stable enough to compare anything. Run `bench.py`: this is row 0 of
   your table.
2. **Cheap wins without retraining.** `channels_last`, `torch.compile`, resolution 160 px (measure both accuracy and
   speed; then fine-tune at 160 px and measure again). Dynamic int8 quantisation: observe that almost nothing changes and
   explain why. Static PTQ int8 (after the from-scratch quantisation of one layer, below) with
   `torchvision.models.quantization.resnet50` or the torchao path (fuse, calibrate, convert): expect a large speed-up
   on x86 and about 4× smaller weights.
3. **One technique that needs training.** Choose one: (a) structured channel pruning (Torch-Pruning, L1 norm, 30–50 %
   of the channels) followed by fine-tuning; (b) distillation from your ResNet-50 into ResNet-18 or MobileNetV3, with
   the control student trained on hard labels; (c) low-rank factorisation of the widest convolutions and the fc layer,
   followed by fine-tuning. Same fine-tuning budget across variants. Run the baseline-vs-variant comparison with three
   seeds (README requirement 5); PTQ is deterministic given the calibration set, so vary the calibration set instead.
4. **Combine.** For example the distilled MobileNetV3 plus static int8, or the pruned ResNet-50 plus int8. Measure.
5. **Sensitivity study.** Accuracy against sparsity (10, 30, 50, 70 %), or against the distillation temperature, or the
   per-layer sensitivity to quantisation (quantise one stage at a time).
6. **Pareto plot and analysis.** Record what failed (unstructured pruning gave zero speed-up; QAT needed more epochs than
   you had; the int8 model was slower on the Apple CPU with the wrong engine) as carefully as what worked, with a
   `torch.profiler` per-operator table for the baseline and for the best variant next to it: that table is the
   evidence for the mechanism explanation, not a story.

Compute: fine-tuning and distillation need a GPU (a free Colab or Kaggle session is enough). Do all
**measurements on the CPU** you report, in one session, and repeat the baseline measurement at the end to check that
the machine did not change under you.

## From scratch: quantising one layer

Take one convolution of your fine-tuned ResNet-50, say `layer3[0].conv2`, with weight $w$ of shape
$(C_{\text{out}}, C_{\text{in}}, k, k)$. Compute a per-output-channel scale $s_c = \max |w_c| / 127$ and zero-point
$0$ (symmetric int8), quantise `w_q = clamp(round(w / s), -128, 127)`, dequantise `w_hat = w_q * s`, write `w_hat`
back into the layer, and evaluate: the accuracy delta against fp32 and the maximum absolute weight error. Then do the
same with an asymmetric per-tensor scheme (one scale and one zero-point for the whole tensor,
$x_q = \mathrm{round}(x / s) + z$) and compare both numbers. About fifteen lines. Check your `w_q` against
`torch.quantize_per_channel(w, s, z, axis=0, dtype=torch.qint8).int_repr()` (an exact match is expected; the call is
deprecated and warns, but works on the pinned torch) or against torchao's per-channel weight quantisation. Now the observers, `fuse_modules` and `convert` are not magic: they do this
for every layer, plus the activations, which is where the calibration batches come in.

If you pick distillation, the KD loss from the recap is the second thing you write yourself: ten lines, checked
against `F.kl_div` with the right reduction (`batchmean`) and the $T^2$ factor in place.

## Mandatory analyses

- The **full measurement table**: baseline plus at least three variants from at least two of the four families.
- The **Pareto plot**.
- A **mechanism explanation** per variant, backed by the profiler table: why it is faster or smaller, or why it is not
  ("50 % unstructured sparsity: same dense kernels, 0 % faster; gzip size 0.55×; accuracy −0.8 pp").
- A **`torch.profiler` per-operator table** (top 10 operators by CPU time) for the baseline and for the best variant.
- **Per-class accuracy deltas** for the best variant against the baseline: which classes pay for the compression?
- For any int8 variant: the **agreement rate** between fp32 and int8 predictions on the test split, and a gallery of
  20 images where they disagree (two models with equal accuracy can disagree on 5 % of images).
- **One sensitivity curve** (step 5).
- The **cost of compressing**: fine-tuning epochs and time, calibration time, compile time.
- The reproducible `bench.py`, the hardware description, the thread count, the quantisation backend.
- **W&B** runs for every fine-tuning and distillation run, including the control student.
- **Honest limits**: CPU-only measurements? which backend? which resolution? what did you not try and why?

## Common pitfalls

The pitfalls shared by all topics are in [the README](../README.md#pitfalls-everybody-falls-into), and the timing
rules in [How to time a model](../README.md#how-to-time-a-model); these are specific to compression.

- **Unstructured pruning "speed-ups".** After `prune.remove()` the zeros are baked into dense weights; the kernels run at
  the same speed and `torch.save` writes the same number of bytes. Claiming "4× smaller" from 75 % sparsity without
  sparse storage is wrong.
- **Structured pruning by hand** breaks the residual shapes. Use Torch-Pruning, or prune only the inner channels of the
  bottleneck blocks (`conv1` / `conv2` outputs), which no skip connection touches.
- **No fine-tuning and no BatchNorm re-calibration** after pruning.
- **Quantisation traps**: not fusing conv-BN-ReLU; calibrating on the test split (a leak); quantised operators run on
  CPU only (move the model and the data); wrong engine (`torch.backends.quantized.engine`: `qnnpack` on Apple Silicon,
  `fbgemm` or `x86` on Intel / AMD) gives errors or slowness; in eager mode the residual addition needs
  `nn.quantized.FloatFunctional`, which is why you should start from `torchvision.models.quantization.resnet50` and
  cite it.
- **Distillation without a control student.** Without it you cannot say what the teacher contributed. Also: the student
  must never see the test images, and the teacher's predictions on the training set are the targets, not its accuracy.
- **Resolution mismatch.** Evaluating at 160 px a model fine-tuned at 224 px understates what 160 px can do; fine-tune
  at the resolution you deploy.
- **`torch.compile` recompilation** on new input shapes. Keep shapes fixed; count the compile time as warm-up but report
  it.
- **Speed-up against a badly run baseline.** Report absolute milliseconds and the machine; "3× faster" than an
  unoptimised baseline with one thread is not a result.
- **"At most 2 pp drop" measured on the split you tuned** the threshold, temperature or sparsity on. That is why you
  carved a val split; test once.

## Stretch goals

- **ONNX Runtime**: export, run fp32 and int8 with its quantiser, compare with PyTorch int8 on the same CPU.
- **QAT vs PTQ**: recover the PTQ accuracy loss with a few QAT epochs; is it worth the training time?
- **Lower bit-widths**: int4 weight-only quantisation with `torchao`; where does accuracy collapse?
- **Deploy for real**: a Raspberry Pi, a phone (ExecuTorch), or a browser; report the measured latency there.
- **2:4 semi-structured sparsity** on an Ampere-or-newer GPU with `torch.sparse`: the one case where sparsity gives a
  hardware speed-up.
- **Distillation done right**: the "patient and consistent" recipe (long schedule, strong augmentation, teacher and
  student see the same crops); how far can MobileNetV3 get?
- **FLOPs vs latency**: a scatter plot across your variants; FLOPs are a weak proxy for time, show by how much.

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
