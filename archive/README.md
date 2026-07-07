# Archive

Unmaintained code kept for reference only. Nothing in this folder is part of
the `excel_table_cnn` package, is covered by tests, or is expected to run.

| Folder / file | What it was | Why it's here |
|---|---|---|
| `experimental_model/` | A from-scratch Faster-R-CNN-style detector (hand-written anchor generator, RPN heads, RoIAlign, NMS, detection head with a PBR output). | Superseded by the torchvision-based model in `excel_table_cnn/model/`. It could never learn as written — it computes no RPN loss, so proposals stay random and the loss collapses to ~0. Kept because its PBR head structure is the starting point for the planned paper-faithful PBR module (see the project roadmap). |
| `ml_classification/`, `algorithms/`, `table_detector.py` | A pre-CNN prototype: sklearn per-cell header classifier plus heuristic table-body growing. | Different approach and scope; depends on a private library (`dkslib`) and broken import paths. Kept for historical reference. |
| `excel-cnn-testing.ipynb` | The old Kaggle driver notebook. | Its imports point at module paths that no longer exist. Replaced by `notebooks/train_kaggle_colab.ipynb`. |
| `dl_tensors.ipynb` | Early tensor-building exploration for the header classifier. | Obsolete. |
| `code_snippets.py` | Scratchpad of torchvision API experiments. | Never runnable; reference only. Gitignored (local copy only). |
