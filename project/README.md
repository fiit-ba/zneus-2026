# ZNEUS 2026 · Project (weeks 9–12)

The last four weeks of the course are yours. Instead of a notebook with TODOs you get a problem, a list of public
datasets and a set of requirements. You build a working solution, evaluate it properly, and defend every decision in a
10-minute talk and a one-page report. Use everything you practised in weeks 1–8 (training loops, splits, transfer learning,
W&B, error analysis), this time without the scaffolding.

> **The golden rule still applies.** Every hand-in consists of the code (a commit in your fork) and, for anything that
> trains a model, a public Weights & Biases link. For the project this means: your code in your fork, a **public** W&B
> project with every experiment you mention, a one-page PDF report, and the presentation in week 12.

## The four topics

| # | Topic | In one sentence | Brief |
|:-:|---|---|---|
| 1 | Medical image segmentation | Predict a pixel mask of a lesion, polyp or tumour and measure it with Dice and IoU. | [topics/segmentation.md](topics/segmentation.md) |
| 2 | Sign-language classification that generalises | Recognise fingerspelled letters and prove it works on people and backgrounds the model has never seen. | [topics/classification.md](topics/classification.md) |
| 3 | Road-scene object detection | Fine-tune a detector for cars, pedestrians and traffic signs; report mAP and speed. | [topics/detection.md](topics/detection.md) |
| 4 | Make ResNet-50 small and fast | Prune, quantise, distil or factorise a fine-tuned ResNet-50 and draw the accuracy-vs-latency Pareto plot. | [topics/compression.md](topics/compression.md) |

Every brief has the same structure: goal, why it matters, datasets (with sizes, licences and links), a suggested
baseline and steps, **one piece you implement from scratch**, mandatory analyses, topic-specific pitfalls, stretch goals
and references. The mandatory analyses are the minimum; the stretch goals are ideas, not requirements. The rules that
apply to every topic (data card, sanity checks, splits, seeds, timing, the pitfalls everybody falls into) live on this
page and are not repeated in the briefs.

How to pick: choose the topic whose *question* interests you, not the one that looks easiest. Segmentation and
detection need a GPU for the real runs (a free Colab or Kaggle GPU is enough for the dataset sizes suggested);
classification runs on a laptop CPU at 64 × 64 pixels, but collecting your own photos costs an afternoon; compression
needs a CPU for the measurements and a GPU only for fine-tuning or distillation. Detection is the most expensive topic
in GPU-hours, which is why its mandatory scope is the smallest. If you are unsure, read the "Common pitfalls" section
of two briefs and pick the one whose pitfalls you would enjoy fighting.

## Timeline

| Week | What happens | What you bring / hand in |
|:-:|---|---|
| 9 | **Kick-off.** The four topics are introduced. You choose one, download the data, write the data card, decide on your metric and your splits, and write down a plan. | Topic choice. |
| 10 | **Consultation 1.** Bring a running baseline: the data card, the sanity-check table, the watch batch, the trivial baselines, a first model with a number, first W&B runs. You get feedback on splits, metric and plan. Without the data card and the sanity table we talk about those instead of your model. | Nothing is handed in, but come with results, not with plans. |
| 11 | **Consultation 2.** Bring your improvement iterations, the from-scratch piece, the error analysis and an outline of the talk. | Nothing is handed in. |
| 12 | **Presentations.** 10-minute talk plus questions. | Report PDF, code in your fork, public W&B link. |

Dates, deadlines, whether you work alone or in a team, the hand-in channel and the points are explained at the lab.
If anything is unclear, ask at the lab or write to [william.brach@stuba.sk](mailto:william.brach@stuba.sk).

Suggested pacing: by the end of week 9 the data are on your disk, the data card is written and you have a plan; by
consultation 1 you have a baseline with a number and the sanity checks pass; by consultation 2 you have at least one
improvement iteration with an analysis; the last week is for the report and the talk, not for training.

## General requirements (all topics)

1. **Reproducibility.** Your project folder has a `README.md` that says how to run everything (exact commands), where
   the data come from and how to download them, which seeds you used, which environment (the repository's
   `pyproject.toml` plus whatever you added), which hardware, and how long training took.
2. **Look at the data first.** Before the first training run, write a **data card**: a notebook section (or a short
   markdown file) with 20 random samples and their labels, masks or boxes; the class or foreground balance; the image
   size distribution; a duplicate check (an md5 of every file, plus a perceptual hash or a comparison of 8 × 8
   thumbnails for near-duplicates); and the oddities you found (empty masks, missing labels, corrupt files, a label
   mapping that does not match the folder names). Each brief says what to look for in its data. This is an hour of
   work, and it is where most projects are won or lost.
3. **Proper splits.** Train / validation / test. You tune on validation and touch the test split once, at the end.
   Where samples are grouped (slices of one patient, frames of one video, photos of one person) the groups do not cross
   splits. If the dataset ships official splits, use them and say so. Write the three file lists to
   `splits/train.txt`, `splits/val.txt` and `splits/test.txt` in your project folder (file names, not data) so that
   every experiment, and the reader, uses the same split. The report says how you split.
4. **Baseline first.** Start with the simplest thing that produces a number: a trivial predictor (majority class, mean
   mask, all-background) and then a simple model. Every later result is compared against it.
5. **Sanity checks, before any real run.** Each one is a cell whose output you keep, and a row in a small table in
   your README (`check | expected | observed | ok?`).
   - *Loss at init.* Write down the expected value **before** you run. Binary cross-entropy on a pixel map is about
     0.69 per pixel at a zero bias (lower if you initialise the bias to the foreground prior); cross-entropy over 24
     classes is ln 24 ≈ 3.18; for a torchvision detector say which of its four loss terms you expect to dominate.
     If the number is off, your labels, your normalisation or your head are wrong, not your architecture.
   - *Overfit one batch.* Eight images, a few hundred steps: Dice above 0.99, 100 % training accuracy, or box losses
     near zero. If the model cannot memorise eight images, the pipeline is broken.
   - *Input-independent baseline.* Train once with the inputs zeroed (or the labels shuffled). It must land on the
     trivial baseline. If it does better, something leaks: the split, the file names, the image size.
   - *Watch batch.* Pick eight fixed validation images in week 9 and log their predictions to W&B as images every
     epoch. You will see the moment the U-Net stops predicting all-background, and the moment it starts to overfit.
   - *Seeds and repeats.* One `SEED` sets `torch.manual_seed`, `numpy`, `random` and the DataLoader generator. Run
     your headline comparison with **three seeds** and report mean ± std. With 150 test images a difference of one
     percentage point is noise; say so instead of calling it a result.
   - *Be the model* (recommended). Label 50 test items yourself, without looking at the answer, and report your own
     accuracy or Dice. Nothing teaches what a metric measures faster.
6. **One piece from scratch.** Each brief names one component (the Dice loss, average precision, Grad-CAM, the
   quantisation of one layer) that you implement in plain `torch` and check against the library on the same input.
   It is 10 to 40 lines, it is assessed, and "the library did it" is not accepted for that piece.
7. **At least one improvement iteration.** Hypothesis ("the model overfits the background, so background augmentation
   should close the gap"), experiment, result, conclusion. Then the next one. Two well-analysed iterations beat ten
   unexplained runs.
8. **Error analysis.** Look at the worst predictions and say what they have in common. A gallery of the worst 20 test
   items is the minimum. Numbers tell you *that* the model fails; the examples tell you *why*.
9. **Weights & Biases for every experiment.** Project `zneus-2026` (or a project of its own, as long as it is public),
   one run per experiment, run names `project-<topic>-<experiment>` (for example `project-seg-unet-scratch-seed0`),
   the configuration logged with `wandb.init(config=...)`. Every number in your report and your slides is traceable to
   a run.
10. **Cite what you reuse.** Datasets, pretrained weights, libraries, code snippets, papers. Reused code is fine; hidden
    reused code is not.
11. **Be honest about what did not work.** A negative result with an explanation is a result. Silently dropping it
    always costs you.

**Where the code lives.** Create a folder `project/<topic>-<your-ais-login>/` in your fork (for example
`project/compression-xnovak/`). Pulling upstream updates then never conflicts with your work. Keep data out of git;
the repository's `.gitignore` already ignores `data/`.

```
project/<topic>-<login>/
  README.md            how to run, data source, seeds, hardware, training time, the sanity-check table
  splits/              train.txt, val.txt, test.txt (file names only)
  notebooks/ or src/   your code; notebooks with outputs are fine
  report.pdf           the one-page report
```

**Extra packages.** `pyproject.toml` contains what weeks 1–8 needed. Each brief lists what its baseline needs on top
(`segmentation_models_pytorch`, `torchmetrics`, `mediapipe`, `torch-pruning`, ...). Add them in your fork with
`uv add <package>`; on a GPU build use `uv add --no-sync <package>` followed by your GPU sync command from
[SETUP.md](../labs/week_01_setup/SETUP.md), otherwise `uv add` puts you back on the CPU build. Commit the changed `pyproject.toml` and
`uv.lock`. In Colab, `!pip install <package>` in the setup cell. Whatever you add is part of the environment
description in your README.

**Compute.** A free Colab or Kaggle GPU session is enough for every topic. Develop on a small subset on your laptop and
run the real experiments on a GPU session; save checkpoints, because free sessions time out.

## Shared protocols

### How to time a model

Every latency number in every topic is measured the same way, and the briefs refer to this section instead of
repeating it.

- `model.eval()`, inside `torch.inference_mode()`, **batch size 1**, one fixed input size (say which).
- CPU: `torch.set_num_threads(k)` with $k$ fixed and reported; report the CPU model and core count. GPU:
  `torch.cuda.synchronize()` before and after the timed call; report the GPU. Never time on MPS or CUDA and call it CPU
  latency.
- 10 warm-up runs, then at least 50 timed runs with `time.perf_counter()` (or `torch.utils.benchmark.Timer`). Report
  the **median** and the p90 in milliseconds.
- Everything in one session, on one machine, with nothing else running. Repeat the first measurement at the end to
  check that the machine did not change under you (Colab CPUs differ between sessions).
- Batch sizes above 1 measure throughput, not latency. Report throughput separately, if at all.
- `torch.compile` and the first CUDA kernels count as warm-up; report the compile time anyway.
- Put this into one function or script (`bench.py`) and run every variant through the same code.

### Downloading from Kaggle

The API key set-up is in [SETUP.md](../labs/week_01_setup/SETUP.md#kaggle). `data/` is git-ignored.

```bash
uv run kaggle datasets download -d <owner>/<dataset> -p data/<name> --unzip   # datasets
uv run kaggle competitions download -c <competition> -p data/<name>           # competitions: accept the rules on the website first
```

### Pitfalls everybody falls into

The briefs list the pitfalls of their own topic. These apply to all four:

- **Evaluating on the training images**, or on the validation images you selected the epoch with.
- **Reporting the best validation epoch as the test result.** Choose the checkpoint on val, evaluate it on test once.
- **Tuning anything on the test split**: a threshold, post-processing, a temperature, a sparsity level.
- **Forgetting `model.eval()`** at inference. Dropout and BatchNorm behave differently in train mode, and torchvision
  detectors return losses instead of boxes.
- **Augmentation left on at evaluation time**, unless it is deliberate test-time augmentation and reported as such.
- **Comparing models trained with different budgets** (epochs, data, resolution, augmentation) and calling the
  difference "architecture".
- **Normalisation mismatch** with a pretrained backbone. Use the mean and std the weights were trained with, and the
  same preprocessing function at training and test time.
- **Timing on a GPU without `torch.cuda.synchronize()`**, or with batch size above 1. See "How to time a model".
- **A test set too small for the claim.** Report $n$, and for accuracy-type metrics a 95 % Wilson interval. A 3 %
  difference with a ± 5 % interval is not a result.
- **A single seed.** See requirement 5.

## Deliverables

All four are required.

1. **Code in your fork.** A commit (or a tag) that you hand in as a link. Notebooks with outputs are fine; a `README.md`
   with run instructions and the sanity-check table is mandatory (see above).
2. **Public W&B link.** The project, or a W&B report / workspace, that shows every run you refer to.
3. **One-page PDF report.** Exactly one A4 page; what goes in it is explained at the lab.
4. **10-minute presentation** in week 12. Free form (slides, live demo, notebook), but it has to show
   - **what** you did (problem, data, model, metric),
   - **how** you proceeded (baseline, iterations, splits),
   - **why** you made each decision (and what the alternatives were),
   - **what worked and what did not**, with evidence (numbers from W&B, plots, example predictions),
   - and it has to **justify** every claim you make.

A structure that fits in 10 minutes (use it, or design your own):

| Minutes | Content |
|:-:|---|
| 0–1 | Problem, data, metric. One slide. |
| 1–3 | Method and the decisions behind it: baseline, splits, model choice, the improvement iterations you planned. |
| 3–6 | Results: the main table and one plot, the comparison against the baseline, a few qualitative examples. |
| 6–8 | What did not work and why; error analysis of the worst cases. |
| 8–9 | Conclusion and what you would do next. |
| 9–10 | Buffer. Ten minutes is short; rehearse with a timer. |

Questions follow the talk.

## Assessment criteria

Points and weights are explained at the lab. Qualitatively, this is what a good project looks like:

- **Problem framing and data handling.** The metric fits the task, the splits are correct and leakage-free, and the
  data card shows you looked at the data (balance, sizes, duplicates, oddities).
- **Method and justification.** A baseline exists, the sanity-check table is in the README and every row passes, the
  from-scratch piece works and matches the library; every non-trivial decision (architecture, loss, augmentation,
  resolution, hyperparameters) comes with a reason and, ideally, with an experiment that supports it.
- **Experimental rigour.** Comparisons are fair (same data, same budget, same test split), seeds are fixed and the
  headline comparison has three of them, runs are logged, the numbers in the report match W&B.
- **Analysis and understanding.** The error analysis is specific, failures are explained, the mandatory analyses of the
  topic brief are present, negative results are reported.
- **Communication.** The talk fits 10 minutes and answers what / how / why; the report fits one page and stands on its
  own; questions are answered with evidence.
- **Code and reproducibility.** Someone else can run it from the README; the structure is clean; reused code and data
  are cited.

Impressive final numbers are not on this list on purpose. A well-analysed Dice of 0.80 beats an unexplained 0.85.

## FAQ

**Can I use pretrained models?** Yes, and for three of the four topics you should. Cite the weights (which model,
trained on what, downloaded from where) and make clear which parts you trained yourself.

**Can I use libraries such as `segmentation_models_pytorch`, `torchmetrics`, `albumentations`, `timm`, `ultralytics`?**
Yes, but you must be able to explain what they do for you (what the loss computes, how the metric is defined, what the
augmentation changes), and the report must cite them. "The library did it" is not an explanation. One piece per topic
you write yourself and check against the library (requirement 6); for the rest, re-implementing what a library offers
is a fine learning exercise, say which parts are yours.

**Can I change the topic after week 9?** Ask at the first consultation. The earlier, the better; after consultation 2
it is too late.

**Can I use a dataset that is not in the brief?** Probably, if it fits the topic and is public with a licence that
allows this use. Ask at the first consultation or earlier, and describe it in the report the way the listed datasets
are described (size, task, licence, link).

**Do I have to reach the suggested target (3× faster, a given accuracy, ...)?** No. The targets exist to make you
argue about trade-offs. A project that misses the target and explains why, with evidence, is a good project. A project
that hits the target with a leaked test set is not.

**My results are bad. Am I in trouble?** Not if you understand why. Report what you tried, show the evidence, explain
the failure and propose what you would try next.

**My baseline already works. Do I still have to do the sanity checks?** Yes. They take an hour, they go into the
README as a table, and they are the first thing we look at in consultation 1. A working baseline with an unexplained
loss at init has a bug you have not found yet.

**Can I use ChatGPT, Copilot or similar tools?** Ask at the lab for the exact policy. Whatever it is, you must be able
to explain every line you hand in, and the assessment above is about your understanding.

**Where do I get a GPU?** Google Colab or Kaggle, both free; see [SETUP.md](../labs/week_01_setup/SETUP.md#google-colab).

**How is the report handed in?** As `report.pdf` in your project folder in the fork.

**Something else is unclear?** Ask at the lab or write to [william.brach@stuba.sk](mailto:william.brach@stuba.sk).

## Links

- Topic briefs: [segmentation](topics/segmentation.md) · [classification](topics/classification.md) ·
  [detection](topics/detection.md) · [compression](topics/compression.md)
- Material you will reuse: [week 8, transfer learning](../labs/week_08_transfer_learning/README.md) (fine-tuning a
  pretrained torchvision model on Oxford-IIIT Pets, W&B logging) and [week 7](../labs/week_07_cifar10_cnn/README.md)
  (the CNN you wrote yourself)
- Andrej Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/): the
  sanity checks above are its first half, condensed
- [Weights & Biases quickstart](https://docs.wandb.ai/quickstart) and [W&B Reports](https://docs.wandb.ai/guides/reports)
  (a public report is a good way to hand in "the W&B link")
