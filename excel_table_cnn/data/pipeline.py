"""End-to-end dataset preparation: download -> annotations -> featurized,
cached samples ready for ``SpreadsheetDataset``.

Workbooks are read natively (.xlsx/.xlsm via openpyxl, .xls via xlrd) — no
LibreOffice required. Pass ``use_libreoffice=True`` to convert legacy files
to .xlsx first instead (needed only for .xlsb, or to compare featurization
fidelity between the two paths).

Featurizing a sheet is the expensive step, so every (file, sheet) tensor is
cached on disk keyed by the file content hash, the sheet name and the
featurization version — reruns and notebook restarts hit the cache.
"""

import hashlib
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .converter import convert_files
from .features import (
    DEFAULT_MAX_COLS,
    DEFAULT_MAX_ROWS,
    FEATURE_NAMES,
    FEATURES_VERSION,
)
from .loader import DatasetLoader
from .markup import MarkupLoader
from .workbook import NATIVE_EXTENSIONS, WorkbookReader
from ..training.dataset import parse_table_range

logger = logging.getLogger(__name__)


def _cache_key(file_path: str, sheet_name: str) -> str:
    hasher = hashlib.sha1()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    hasher.update(sheet_name.encode("utf-8"))
    hasher.update(FEATURES_VERSION.encode("utf-8"))
    return hasher.hexdigest()


def _make_sample(
    array: np.ndarray, table_ranges: List[str], file_path: str, sheet_name: str
) -> Dict:
    return {
        "tensor": torch.from_numpy(array).permute(2, 0, 1).contiguous(),
        "boxes": torch.tensor(
            [parse_table_range(tr) for tr in table_ranges], dtype=torch.float32
        ).reshape(-1, 4),
        "file_path": file_path,
        "sheet_name": sheet_name,
        "feature_names": list(FEATURE_NAMES),
    }


def build_sheet_sample(
    file_path: str,
    sheet_name: str,
    table_ranges: List[str],
    cache_dir: Optional[str] = None,
    array: Optional[np.ndarray] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: int = DEFAULT_MAX_COLS,
) -> Dict:
    """Featurize one sheet (or load it from cache) into a sample dict.

    ``array`` may carry a pre-featurized sheet to avoid reopening the file.
    """
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, _cache_key(file_path, sheet_name) + ".pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=True)

    if array is None:
        with WorkbookReader(file_path) as reader:
            array = reader.sheet_array(sheet_name, max_rows=max_rows, max_cols=max_cols)

    sample = _make_sample(array, table_ranges, file_path, sheet_name)
    if cache_path is not None:
        torch.save(sample, cache_path)
    return sample


def build_samples(
    files_df,
    data_folder_path: str,
    cache_dir: Optional[str] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: int = DEFAULT_MAX_COLS,
) -> List[Dict]:
    """Build samples for every (file, sheet) row of a files dataframe.

    Expects columns: parent_path, file_name, sheet_name, table_range (list of
    Excel range strings). Sheets that fail to load are logged and skipped.
    """
    from tqdm import tqdm

    samples: List[Dict] = []
    if files_df.empty:
        return samples
    errors = 0
    grouped = files_df.groupby(
        files_df.apply(lambda r: os.path.join(r["parent_path"], r["file_name"]), axis=1)
    )
    for rel_path, group in tqdm(grouped, desc="Featurizing sheets"):
        file_path = os.path.join(data_folder_path, rel_path)
        reader = None
        try:
            for _, row in group.iterrows():
                sheet_name = row["sheet_name"]
                cached = None
                if cache_dir is not None and os.path.exists(file_path):
                    cache_path = os.path.join(
                        cache_dir, _cache_key(file_path, sheet_name) + ".pt"
                    )
                    if os.path.exists(cache_path):
                        cached = torch.load(cache_path, weights_only=True)
                if cached is not None:
                    samples.append(cached)
                    continue
                if reader is None:
                    reader = WorkbookReader(file_path).__enter__()
                samples.append(
                    build_sheet_sample(
                        file_path, sheet_name, list(row["table_range"]),
                        cache_dir=cache_dir,
                        array=reader.sheet_array(sheet_name, max_rows=max_rows,
                                                 max_cols=max_cols),
                        max_rows=max_rows, max_cols=max_cols,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - corpus files are wild
            errors += 1
            logger.error("Skipping %s: %s", file_path, exc)
        finally:
            if reader is not None:
                reader.close()
    if errors:
        logger.warning("Skipped %d file(s) due to errors", errors)
    return samples


def get_train_test(
    data_folder_path: str = "./data",
    dataset_name: str = "VEnron2",
    markup_name: str = "tablesense",
    train_size: Optional[int] = None,
    testing_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: int = DEFAULT_MAX_COLS,
    use_libreoffice: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """Download the corpus + annotations and return (train, test) sample lists.

    ``train_size``/``testing_size`` subsample sheets (None = use all).
    ``cache_dir`` defaults to ``<data_folder_path>/feature_cache``.
    ``use_libreoffice=True`` converts legacy files to .xlsx before
    featurization (requires LibreOffice on PATH); the default reads .xls
    natively via xlrd.
    """
    if cache_dir is None:
        cache_dir = os.path.join(data_folder_path, "feature_cache")

    logger.info("Ensuring dataset %s is available...", dataset_name)
    dataset_loader = DatasetLoader(save_path=data_folder_path)
    dataset_loader.get_dataset(dataset_name)
    dataset_files = dataset_loader.get_files(dataset_name)

    logger.info("Loading markup %s...", markup_name)
    markup_files = MarkupLoader().get_markup(markup_name)

    # Annotations may reference .xls while the corpus (after conversion)
    # holds .xlsx — match on extension-less names.
    dataset_files["file_name_no_ext"] = dataset_files["file_name"].apply(
        lambda x: os.path.splitext(x)[0]
    )
    markup_files["file_name_no_ext"] = markup_files["file_name"].apply(
        lambda x: os.path.splitext(x)[0]
    )
    files_df = markup_files.merge(
        dataset_files, how="inner", on=["file_name_no_ext", "parent_path"]
    )
    files_df = files_df.drop(columns=["file_name_x", "file_name_no_ext"])
    files_df = files_df.rename(columns={"file_name_y": "file_name"})

    train_df = files_df[files_df["set_type"] == "training_set"]
    test_df = files_df[files_df["set_type"] == "testing_set"]
    if train_size is not None:
        train_df = train_df.sample(min(train_size, len(train_df)), random_state=seed)
        if train_df.empty:
            train_df = train_df.reindex(columns=files_df.columns)
    if testing_size is not None:
        test_df = test_df.sample(min(testing_size, len(test_df)), random_state=seed)

    if use_libreoffice:
        train_df = convert_files(train_df, data_folder_path)
        test_df = convert_files(test_df, data_folder_path)
    else:
        for name, df in (("train", train_df), ("test", test_df)):
            unsupported = ~df["file_name"].str.lower().str.endswith(NATIVE_EXTENSIONS)
            if unsupported.any():
                logger.warning(
                    "Dropping %d %s file(s) with no native reader (e.g. .xlsb); "
                    "rerun with use_libreoffice=True to include them",
                    int(unsupported.sum()), name,
                )
        train_df = train_df[
            train_df["file_name"].str.lower().str.endswith(NATIVE_EXTENSIONS)
        ]
        test_df = test_df[
            test_df["file_name"].str.lower().str.endswith(NATIVE_EXTENSIONS)
        ]

    train_samples = build_samples(
        train_df, data_folder_path, cache_dir=cache_dir,
        max_rows=max_rows, max_cols=max_cols,
    )
    test_samples = build_samples(
        test_df, data_folder_path, cache_dir=cache_dir,
        max_rows=max_rows, max_cols=max_cols,
    )
    logger.info("Built %d train / %d test samples", len(train_samples), len(test_samples))
    return train_samples, test_samples
