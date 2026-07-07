# ExcelTableCNN

Open-source table detection on Excel sheets with computer vision: a spreadsheet
is featurized into an image-like tensor (one channel per cell feature) and a
Faster R-CNN detector predicts the bounding ranges of every table on the sheet.

This is an **independent open-source reimplementation inspired by the
TableSense paper** — Dong et al., *TableSense: Spreadsheet Table Detection with
Convolutional Neural Networks*, AAAI 2019
([arXiv:2106.13500](https://arxiv.org/abs/2106.13500)). It is not affiliated
with or endorsed by Microsoft; no code from any Microsoft repository is used.

```python
from excel_table_cnn import detect_tables

detect_tables("report.xlsx", weights="checkpoints/final.pt")
# [{"sheet": "Q3", "range": "B2:H45", "score": 0.97},
#  {"sheet": "Q3", "range": "J2:M10", "score": 0.91}]
```

**Status (alpha):** the pipeline is functional and tested end-to-end (see
[Testing](#testing)); pre-trained weights are not published yet, so to get
useful detections you currently need to train first — see
[Training](#training).

## How it works

1. **Featurization** (`excel_table_cnn/data/features.py`): every cell of a
   sheet becomes a 30-dimensional feature vector following the paper's
   scheme — emptiness, string content and statistics (length, digit/letter
   ratios, `%`/decimal presence), number-format template classification
   (numeric/date/time), merge membership and direction, bold/italic font,
   the four borders, fill and non-default fill/font colors, alignment,
   wrapped text, indentation, and formula presence. A sheet becomes an
   `H×W×30` tensor at cell resolution. Trailing all-default rows/columns are
   trimmed and the used range is capped (default `2048×512`) to survive
   sheets with stray formatting.
2. **Detection** (`excel_table_cnn/model/`): a stride-1 fully convolutional
   backbone (no pooling — cell-level resolution is preserved, as in the
   paper) feeds a torchvision Faster R-CNN with anchors sized in cell units
   — the anchor lattice is tuned on the annotated corpus's box-shape census
   (spreadsheet tables are tall: median height/width ratio 2.5, p95 = 26) —
   and a transform that skips image-style resizing/normalization.
3. **Grid-context backbone** (`excel_table_cnn/model/grid_context.py`) —
   *this project's own addition over TableSense*: table boundaries are
   global row/column events, so the backbone (a) receives row/column
   fill-density and coordinate priors as derived channels, (b) uses dilated
   convolutions to span typical table heights, and (c) runs an axial
   strip-pooling block that gives every cell a learned summary of its entire
   row and column.
4. **PBR boundary snapping** (`excel_table_cnn/model/pbr.py`): the paper's
   precise-bounding-box-regression idea, discretized — for each detected
   edge, a head reads a ±7-cell feature band around it and *classifies* the
   integer offset to the true boundary, directly optimizing exact-boundary
   (EoB-0) accuracy. Trained by recovering jittered ground-truth boxes.
5. **Boxes ↔ ranges** (`excel_table_cnn/training/dataset.py`): boxes use a
   half-open cell convention — `"A1:C3"` ↔ `[0, 0, 3, 3]` — so even a
   single-cell table has positive area. `box_to_range()` converts predictions
   back to Excel ranges.
6. **Evaluation** (`excel_table_cnn/evaluation/`): the paper's
   Error-of-Boundary metric. EoB of a detection is the maximum absolute
   boundary deviation in cells; a detection counts as correct at EoB-0
   (exact) or EoB-2 (≤ 2 cells off). This is far stricter than IoU and is
   the metric that matters for downstream extraction.

## Repository structure

```
excel_table_cnn/
  __init__.py            # public API (detect_tables, build_model, train_model, ...)
  device.py              # device resolution: auto = CUDA -> CPU; MPS opt-in
  data/
    loader.py            # corpus download (VEnron2 etc. from figshare)
    markup.py            # TableSense table-range annotations (O-UDA licensed)
    workbook.py          # format dispatch: .xlsx/.xlsm (openpyxl), .xls (xlrd)
    features.py          # cell featurization -> (H, W, 30) tensors (openpyxl)
    features_xls.py      # same channels for legacy .xls via xlrd (no LibreOffice)
    converter.py         # optional .xlsb/.xls -> .xlsx via headless LibreOffice
    census.py            # GT box-shape stats + anchor-lattice coverage
    pipeline.py          # end-to-end dataset build with on-disk tensor caching
  model/
    backbone.py          # stride-1 FCN backbone (GroupNorm; batch size is 1)
    grid_context.py      # NOVEL: row/col priors + axial strip pooling
    pbr.py               # PBR boundary snapping (per-edge offset classification)
    rcnn.py              # customized torchvision Faster R-CNN, corpus-tuned anchors
    detector.py          # TableDetectionModel + build_model()
  training/
    dataset.py           # box convention, SpreadsheetDataset, validation
    train.py             # trainer (per-loss logging, warmup, AMP, checkpoints) + CLI
  evaluation/
    eob.py               # EoB metric, matching, precision/recall
    evaluate.py          # evaluation harness + report formatting
  inference.py           # detect_tables() / load_model()
tests/                   # unit tests + the M0 overfit smoke test (see below)
notebooks/
  train_kaggle_colab.ipynb  # training runbook for Kaggle / Colab
archive/                 # unmaintained legacy code, kept for reference only
```

## Installation

Requires Python ≥ 3.10. The project is managed with [uv](https://docs.astral.sh/uv/)
(standard `pyproject.toml` + committed `uv.lock` — no `requirements.txt`):

```bash
git clone https://github.com/Flagro/ExcelTableCNN.git
cd ExcelTableCNN
uv sync --extra dev   # creates .venv with locked, reproducible deps
uv run pytest -m "not slow"
```

Plain pip works too (Kaggle/Colab, or if you don't use uv):

```bash
pip install -e .          # library
pip install -e ".[dev]"   # library + test tooling
```

On CPU-only Linux machines you can save gigabytes by installing the CPU torch
wheels first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

**LibreOffice is optional.** `.xlsx`/`.xlsm` are read with openpyxl and legacy
`.xls` natively with xlrd (including formatting — the one caveat: the
`formula` feature channel is always 0 for `.xls`, since xlrd only exposes
cached formula results). Only `.xlsb` files require converting via
LibreOffice (`--use-libreoffice` / `use_libreoffice=True`).

## Quickstart: inference

```python
from excel_table_cnn import detect_tables, load_model

model = load_model("checkpoints/final.pt")           # a trainer checkpoint
tables = detect_tables("report.xlsx", model=model)    # all sheets
tables = detect_tables("report.xlsx", sheet_name="Q3", model=model,
                       score_threshold=0.7)           # one sheet, stricter
```

Each detection is `{"sheet": str, "range": "B2:H45", "score": float}`.

## Training

Training data: the [VEnron2 corpus](https://figshare.com/articles/dataset/Venron/5714233)
(spreadsheets extracted from the Enron email dump) with table-range
annotations from the
[Microsoft TableSense repository](https://github.com/microsoft/TableSense)
(licensed under the Open Use of Data Agreement v1.0). Both are downloaded on
demand — nothing is redistributed in this repository.

### Locally

The console script drives the whole pipeline — download, conversion, cached
featurization, training, evaluation:

```bash
excel-table-cnn-train --data-dir ./data --train-size 50 --test-size 20 \
    --epochs 20 --checkpoint-dir ./checkpoints
```

- `--train-size/--test-size` subsample sheets for quick runs; drop them to
  train on everything.
- No LibreOffice needed: `.xls` files are read natively. Add
  `--use-libreoffice` only if you want the conversion path (e.g. for `.xlsb`).
- Devices: the default `--device auto` picks CUDA when available, otherwise
  CPU; mixed precision turns on automatically with CUDA (`--no-amp` to
  disable). See [Apple Silicon](#apple-silicon-macs) below for MPS.
- Checkpoints: `last.pt` (every epoch) and `final.pt`; reload with
  `excel_table_cnn.load_checkpoint(path)`.
- Feature tensors are cached under `<data-dir>/feature_cache/` keyed by file
  hash — the expensive featurization step runs once per sheet, ever.

Two companion commands work with the artifacts the trainer produces:

```bash
# EoB report for a checkpoint on the annotated test split (+ worst offenders):
excel-table-cnn-eval --weights checkpoints/final.pt --data-dir ./data --worst 5

# Detect tables in any spreadsheet from the command line:
excel-table-cnn-detect report.xls --weights checkpoints/final.pt
# Sheet1!B2:H45   score=0.973
```

The same is available as a library via `get_train_test()`,
`SpreadsheetDataset`, `build_model()`, `train_model()` — see the notebook for
the exact sequence.

If figshare refuses the automatic corpus download (it sometimes answers
non-browser clients with an empty `202`), download the archive in a browser
and drop it at `<data-dir>/VEnron2.7z` — the loader verifies its MD5 against
the figshare-published checksum and unpacks it.

### On Kaggle

1. Create a notebook, enable a GPU accelerator, internet on.
2. Upload/import [`notebooks/train_kaggle_colab.ipynb`](notebooks/train_kaggle_colab.ipynb)
   and run it top to bottom. It pip-installs this repo, downloads the corpus
   into `/kaggle/working/data`, trains (CUDA + AMP picked up automatically),
   and prints an EoB report.

Shell equivalent inside any Kaggle cell:

```bash
%pip install -q git+https://github.com/Flagro/ExcelTableCNN.git
!excel-table-cnn-train --data-dir /kaggle/working/data --epochs 20
```

### On Colab

Same notebook works. One Colab-specific note: mount Google Drive and point
`--data-dir`/`--checkpoint-dir` at it if you want the feature cache and
checkpoints to survive the session.

### Apple Silicon Macs

Everything runs locally out of the box — `pip`/`uv` install the MPS-capable
torch wheels on macOS, and `.xls` reading needs no LibreOffice. Two notes:

- The default `--device auto` uses the **CPU** on Macs, deliberately: with
  the current small backbone and batch size 1, measured training throughput
  on MPS is 2–5× *slower* than on the M-series performance cores (detection
  heads launch many tiny GPU kernels; overhead dominates).
- MPS is still fully supported and covered by a test
  (`tests/test_mps_smoke.py`) — pass `--device mps` (or `device="mps"`) to
  use it. Worth re-benchmarking when the backbone grows (see roadmap).

### Diagnosing a run

The trainer logs all four detection losses separately every `log_every`
steps: `loss_objectness` and `loss_rpn_box_reg` (region proposal network),
`loss_classifier` and `loss_box_reg` (detection head). A healthy run has all
four non-zero and trending down. If `loss_objectness`/`loss_classifier` sit
near zero from the first steps while detections are garbage, the model is
classifying everything as background — a data/label problem, not a tuning
problem (this exact failure mode is why the logging exists).

## Evaluation

```python
from excel_table_cnn import SpreadsheetDataset, evaluate_model, format_report

report = evaluate_model(model, SpreadsheetDataset(test_samples), device="cuda")
print(format_report(report))
# Evaluated 20 sheets, 34 tables:
#   EoB-0: precision=0.61 recall=0.55 (tp=..., fp=..., fn=...)
#   EoB-2: precision=0.78 recall=0.71 (...)
```

`report["per_sheet"]` carries per-sheet predictions, ground truth, and the
best EoB achieved per table — the starting point for error analysis. For
reference, the TableSense paper reports EoB-2 recall 91.3% / precision 86.5%,
trained on 10,220 hand-labeled sheets; this project trains on the much
smaller VEnron2 annotation set, so expect substantially lower numbers.

Measured baseline (2026-07): a 16-minute CPU run (239 training sheets,
40 epochs, `--max-rows 512 --max-cols 128`) reaches **EoB-2 recall 23.9% /
precision 41.0%** on a held-out 30-sheet VEnron2 test split, with EoB-0
at zero — exact boundaries are what the roadmap's PBR head is for. Treat
this as the floor: more sheets, more epochs, and the Phase-4 accuracy work
all remain on the table.

## Testing

```bash
uv run pytest -m "not slow"   # unit tests, a few seconds
uv run pytest                 # + slow tests: M0 overfit gate (~1 min CPU) and,
                              #   on Macs, the MPS compatibility smoke test
```

The suite covers featurization for both backends (every channel asserted
against crafted .xlsx *and* .xls workbooks, plus a channel-parity test
between them), the box convention (including the degenerate
single-cell/column regressions), dataset validation, model construction and
loss components, the PBR head (including a module-level "learns to snap
edges" test), the grid-context blocks, the box census, device resolution,
the EoB metric, the evaluation harness, the feature cache, the training
loop, and inference output. The slow overfit test is the project's core
gate: the model must overfit a single synthetic sheet and reproduce its
table at **EoB-0 (cell-exact)** — proving the whole pipeline, PBR snapping
included, can learn. CI runs both jobs on every push.

## Roadmap

- **Done:** correct data pipeline (30-channel featurization, box convention,
  caching), torchvision-based detector with corpus-tuned anchors, PBR
  boundary snapping, the grid-context backbone, per-component training
  diagnostics, EoB evaluation, test suite with the EoB-0 overfit gate.
- **Next — accuracy:** train on the full annotation set on GPU,
  augmentation, categorical color encoding, iterative PBR refinement,
  segmentation branch.
- **Then — release:** published pre-trained weights + model card, PyPI
  package.

## Licensing & attribution

- Code: [MIT](LICENSE), © Anton Potapov.
- The TableSense **method** is reimplemented from the published paper; please
  cite the paper if you build on this:

  ```bibtex
  @inproceedings{dong2019tablesense,
    title={TableSense: Spreadsheet Table Detection with Convolutional Neural Networks},
    author={Dong, Haoyu and Liu, Shijie and Han, Shi and Fu, Zhouyu and Zhang, Dongmei},
    booktitle={AAAI},
    year={2019}
  }
  ```

- Table-range **annotations** come from the Microsoft TableSense repository
  under the Open Use of Data Agreement v1.0; the **VEnron2 corpus** is
  third-party research data fetched from figshare. Neither is redistributed
  here.
