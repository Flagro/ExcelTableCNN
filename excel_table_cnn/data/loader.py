"""Corpus download from figshare, with integrity verification.

All URLs and checksums below were verified (2026-07-07) against the figshare
API for project 20116 "Versioned Spreadsheet Corpora" (author Wensheng Dou,
all articles CC0) — the upstream source of the annotated corpora.

Note: figshare's downloader sometimes answers non-browser clients with an
empty ``202 Accepted`` response. If the automatic download fails, download
the archive in a browser and drop it at ``<save_path>/<dataset>.7z`` — the
loader picks it up, verifies the MD5, and unpacks it.
"""

import hashlib
import logging
import os
import shutil

import pandas as pd
import requests
from py7zr import unpack_7zarchive

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; ExcelTableCNN dataset loader)"


class DatasetLoader:
    def __init__(self, save_path):
        # Where the datasets will be saved
        self.save_path = save_path

        # figshare download URLs (stable file IDs)
        self.datasets = {
            "VEnron2": "https://figshare.com/ndownloader/files/8639470",
            "VFUSE": "https://figshare.com/ndownloader/files/7889911",
            "VEUSES": "https://figshare.com/ndownloader/files/7889902",
            "VEnron1.1": "https://figshare.com/ndownloader/files/7889866",
            "VEnron1.0": "https://figshare.com/ndownloader/files/7889947",
        }
        # figshare-published MD5s (from api.figshare.com article metadata)
        self.known_md5 = {
            "VEnron2": "9a724c5f667f7fa371619774a1b19c4b",
            "VFUSE": "e822971e2a76e033516b6e6adeb605e7",
            "VEUSES": "46f5b8b4233473b2e2b7d388c56a0ea0",
            "VEnron1.1": "b83818285dea1ea0161ab5aa1b803c81",
            "VEnron1.0": "15a3430526b01a3ace679225a450cc1e",
        }

    def _archive_path(self, dataset_name):
        return os.path.join(self.save_path, f"{dataset_name}.7z")

    @staticmethod
    def _md5(path):
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def verify_archive(self, dataset_name):
        """Check the downloaded archive against the figshare-published MD5."""
        archive = self._archive_path(dataset_name)
        expected = self.known_md5.get(dataset_name)
        if expected is None:
            logger.warning("No known checksum for %s; skipping verification", dataset_name)
            return True
        actual = self._md5(archive)
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for {archive}: got {actual}, expected {expected}. "
                "The download is corrupt or tampered with — delete it and retry."
            )
        logger.info("Checksum OK for %s (%s)", dataset_name, expected)
        return True

    def download_dataset(self, dataset_name):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found.")

        url = self.datasets[dataset_name]
        archive = self._archive_path(dataset_name)
        os.makedirs(self.save_path, exist_ok=True)

        response = requests.get(
            url, stream=True, headers={"User-Agent": USER_AGENT}, timeout=60
        )
        response.raise_for_status()
        with open(archive, "wb") as file:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    file.write(chunk)

        if os.path.getsize(archive) == 0:
            os.remove(archive)
            raise RuntimeError(
                f"figshare returned an empty response for {url} (it sometimes "
                "answers non-browser clients with 202/empty). Download the file "
                f"in a browser and save it as {archive}, then rerun — the loader "
                "will verify and unpack it."
            )
        self.verify_archive(dataset_name)
        logger.info("Dataset %s downloaded successfully.", dataset_name)

    def unpack_dataset(self, dataset_name, keep_archive=False):
        archive = self._archive_path(dataset_name)
        extract_path = os.path.join(self.save_path, dataset_name)

        if not os.path.exists(archive):
            raise FileNotFoundError(f"Dataset archive not found at {archive}.")

        # Register the .7z format with shutil, only if it's not already registered
        if "7zip" not in shutil._UNPACK_FORMATS:
            shutil.register_unpack_format("7zip", [".7z"], unpack_7zarchive)

        shutil.unpack_archive(archive, extract_path, "7zip")
        logger.info("Dataset %s unpacked successfully.", dataset_name)

        if not keep_archive:
            os.remove(archive)
            logger.info("%s.7z removed after unpacking.", dataset_name)

    def get_dataset(self, dataset_name, check_exists=True):
        dataset_dir = os.path.join(self.save_path, dataset_name)

        if check_exists and os.path.exists(dataset_dir):
            logger.info("Dataset %s already exists at %s.", dataset_name, dataset_dir)
            return

        # A manually-downloaded archive (e.g. via browser) is picked up here.
        if os.path.exists(self._archive_path(dataset_name)):
            self.verify_archive(dataset_name)
        else:
            self.download_dataset(dataset_name)
        self.unpack_dataset(dataset_name)

    def get_files(self, dataset_name):
        dataset_dir = os.path.join(self.save_path, dataset_name)

        if not os.path.exists(dataset_dir):
            logger.error("Directory %s does not exist.", dataset_dir)
            return pd.DataFrame(columns=["file_name", "parent_path"])

        result = []
        for root, _dirs, files in os.walk(dataset_dir):
            for file in files:
                relative_path = os.path.relpath(root, self.save_path)
                result.append({"file_name": file, "parent_path": relative_path})

        return pd.DataFrame(result)
