"""Training loop for the table detector.

Every step logs all four Faster R-CNN loss components separately
(``loss_objectness``, ``loss_rpn_box_reg``, ``loss_classifier``,
``loss_box_reg``): a detector that silently stops learning (e.g. no positive
proposals) is visible in the logs instead of hiding inside one aggregate
number.
"""

import argparse
import logging
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SpreadsheetDataset, collate_fn
from ..data.features import FEATURE_NAMES, NUM_FEATURES
from ..device import resolve_amp, resolve_device
from ..model.detector import TableDetectionModel, build_model

logger = logging.getLogger(__name__)

LOSS_KEYS = (
    "loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg",
    "loss_pbr",  # absent when the model is built with use_pbr=False
)


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_steps: int = 100
    lr_step_size: Optional[int] = None  # epochs between 10x LR decays; None = constant
    grad_clip: float = 10.0
    amp: Optional[bool] = None  # None = auto: on for CUDA, off for MPS/CPU
    seed: int = 42
    device: str = "auto"  # cuda -> mps -> cpu
    checkpoint_dir: Optional[str] = None
    log_every: int = 10


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloader(dataset: SpreadsheetDataset, shuffle: bool = True) -> DataLoader:
    # Batch size 1: sheets have wildly different sizes (paper trains the same way).
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=collate_fn)


def save_checkpoint(model: TableDetectionModel, path: str, **extra) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_channels": model.in_channels,
            "num_classes": model.num_classes,
            "config": model.config,
            "feature_names": list(FEATURE_NAMES),
            **extra,
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu", **overrides) -> TableDetectionModel:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    kwargs = {**checkpoint.get("config", {}), **overrides}
    model = build_model(
        checkpoint["in_channels"], num_classes=checkpoint["num_classes"], **kwargs
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def train_model(
    model: TableDetectionModel,
    dataset: SpreadsheetDataset,
    config: TrainConfig,
) -> List[Dict[str, float]]:
    """Train and return the per-step history of loss components."""
    seed_everything(config.seed)
    device = resolve_device(config.device)
    logger.info("Training on device: %s", device)
    if device.type == "cuda":
        # Allow TF32 matmuls on tensor-core GPUs — big speedup, no visible
        # accuracy cost for detection training.
        torch.set_float32_matmul_precision("high")
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=0.1)
        if config.lr_step_size
        else None
    )
    use_amp = resolve_amp(config.amp, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loader = make_dataloader(dataset)
    history: List[Dict[str, float]] = []
    global_step = 0

    for epoch in range(config.epochs):
        epoch_sums: Dict[str, float] = {key: 0.0 for key in LOSS_KEYS}
        epoch_total = 0.0

        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Linear warmup: large detection losses on random weights can
            # otherwise blow up the RPN in the first steps.
            if global_step < config.warmup_steps:
                warmup_factor = (global_step + 1) / config.warmup_steps
                for group in optimizer.param_groups:
                    group["lr"] = config.lr * warmup_factor

            optimizer.zero_grad()
            autocast = torch.amp.autocast("cuda") if use_amp else nullcontext()
            with autocast:
                loss_dict = model(images, targets)
                total_loss = sum(loss_dict.values())

            scaler.scale(total_loss).backward()
            if config.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            record = {key: float(loss_dict[key]) for key in LOSS_KEYS if key in loss_dict}
            record.update(
                epoch=epoch,
                step=global_step,
                loss_total=float(total_loss),
                lr=optimizer.param_groups[0]["lr"],
            )
            history.append(record)
            for key in LOSS_KEYS:
                epoch_sums[key] += record.get(key, 0.0)
            epoch_total += record["loss_total"]

            if config.log_every and global_step % config.log_every == 0:
                components = ", ".join(
                    f"{key.removeprefix('loss_')}={record.get(key, 0.0):.4f}"
                    for key in LOSS_KEYS
                )
                logger.info(
                    "epoch %d step %d: loss=%.4f (%s)",
                    epoch, global_step, record["loss_total"], components,
                )
            global_step += 1

        if scheduler is not None:
            scheduler.step()

        n = max(len(loader), 1)
        components = ", ".join(
            f"{key.removeprefix('loss_')}={epoch_sums[key] / n:.4f}" for key in LOSS_KEYS
        )
        logger.info(
            "Epoch %d/%d: mean loss=%.4f (%s)",
            epoch + 1, config.epochs, epoch_total / n, components,
        )

        if config.checkpoint_dir:
            save_checkpoint(
                model, os.path.join(config.checkpoint_dir, "last.pt"),
                epoch=epoch, mean_loss=epoch_total / n,
            )

    return history


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the ExcelTableCNN table detector on VEnron2."
    )
    parser.add_argument("--data-dir", default="./data", help="dataset/download directory")
    parser.add_argument("--cache-dir", default=None,
                        help="feature cache directory (default: <data-dir>/feature_cache)")
    parser.add_argument("--dataset", default="VEnron2")
    parser.add_argument("--train-size", type=int, default=None,
                        help="subsample N training sheets (default: all)")
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr-step-size", type=int, default=None,
                        help="epochs between 10x LR decays (default: constant LR)")
    parser.add_argument("--device", default="auto",
                        help="cpu, cuda, mps or auto (auto = cuda if available, else cpu)")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="mixed precision; default: on for CUDA, off otherwise")
    parser.add_argument("--use-libreoffice", action="store_true",
                        help="convert legacy files to .xlsx via LibreOffice instead of "
                             "reading .xls natively (required only for .xlsb)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="cap featurized sheet height (cells); lower = faster/less RAM")
    parser.add_argument("--max-cols", type=int, default=None,
                        help="cap featurized sheet width (cells)")
    parser.add_argument("--pbr", action=argparse.BooleanOptionalAction, default=True,
                        help="PBR boundary-snapping head (--no-pbr to ablate)")
    parser.add_argument("--grid-context", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="grid-context backbone (--no-grid-context to ablate)")
    parser.add_argument("--no-eval", action="store_true",
                        help="skip evaluation on the test split after training")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from ..data.features import DEFAULT_MAX_COLS, DEFAULT_MAX_ROWS
    from ..data.pipeline import get_train_test
    from ..evaluation.evaluate import evaluate_model, format_report

    device = resolve_device(args.device)

    train_samples, test_samples = get_train_test(
        data_folder_path=args.data_dir,
        dataset_name=args.dataset,
        train_size=args.train_size,
        testing_size=args.test_size,
        cache_dir=args.cache_dir,
        seed=args.seed,
        use_libreoffice=args.use_libreoffice,
        max_rows=args.max_rows or DEFAULT_MAX_ROWS,
        max_cols=args.max_cols or DEFAULT_MAX_COLS,
    )
    train_dataset = SpreadsheetDataset(train_samples)
    logger.info("Training on %d sheets (skipped %d)",
                len(train_dataset), len(train_dataset.skipped))

    model = build_model(
        in_channels=NUM_FEATURES, use_pbr=args.pbr, use_grid_context=args.grid_context
    )
    config = TrainConfig(
        epochs=args.epochs, lr=args.lr, lr_step_size=args.lr_step_size,
        device=str(device), seed=args.seed,
        amp=args.amp, checkpoint_dir=args.checkpoint_dir,
    )
    train_model(model, train_dataset, config)
    final_path = os.path.join(args.checkpoint_dir, "final.pt")
    save_checkpoint(model, final_path)
    logger.info("Saved final checkpoint to %s", final_path)

    if not args.no_eval and test_samples:
        test_dataset = SpreadsheetDataset(test_samples)
        report = evaluate_model(model, test_dataset, device=str(device))
        print(format_report(report))


if __name__ == "__main__":
    main()
