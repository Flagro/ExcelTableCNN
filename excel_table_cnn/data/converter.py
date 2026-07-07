"""Convert legacy spreadsheet formats to .xlsx via headless LibreOffice.

Openpyxl only reads .xlsx; the VEnron2 corpus ships as .xls. Conversion is
lossy for some formatting details, which is a known limitation — see the
README's dataset section.
"""

import logging
import os
import subprocess

from tqdm import tqdm

logger = logging.getLogger(__name__)

CONVERTIBLE_EXTENSIONS = (".xls", ".xlsb")


def convert_file(file_path: str, output_dir: str) -> bool:
    """Convert one file to .xlsx next to it. Returns True if the .xlsx exists."""
    output_file_path = os.path.splitext(file_path)[0] + ".xlsx"
    if os.path.exists(output_file_path):
        return True
    try:
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "xlsx",
             file_path, "--outdir", output_dir],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.error("Error converting %s: %s", file_path, exc)
    return os.path.exists(output_file_path)


def convert_files(files_df, data_folder_path: str):
    """Convert every .xls/.xlsb row of a files dataframe; returns an updated
    copy where converted rows point at the .xlsx file name."""
    updated_files_df = files_df.copy()

    for index, row in tqdm(files_df.iterrows(), total=files_df.shape[0],
                           desc="Converting files"):
        file_name, file_ext = os.path.splitext(row["file_name"])
        if file_ext.lower() not in CONVERTIBLE_EXTENSIONS:
            continue
        file_path = os.path.join(data_folder_path, row["parent_path"], row["file_name"])
        output_directory = os.path.join(data_folder_path, row["parent_path"])
        if convert_file(file_path, output_directory):
            updated_files_df.at[index, "file_name"] = file_name + ".xlsx"

    return updated_files_df
