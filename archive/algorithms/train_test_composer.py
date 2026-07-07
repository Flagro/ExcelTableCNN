import pandas as pd
import numpy as np
from typing import Optional
from openpyxl.utils import range_boundaries, get_column_letter
from cell_features import get_cell_features, apply_neighbouring_features
from dkslib.excel_io import get_df
from dkslib.excel_io import get_workbook
import enum


class BalanceMode(enum.Enum):
    none = 1
    down_sample = 2


def check_imb(n):
    neg, pos = np.bincount(n)
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))


def get_train_data(train_spec_file=r".\train\Table detection spec.xlsx", export_path=r".\train\train.pkl"):
    df = get_df(train_spec_file)

    train_dfs = []

    for idx, row in df.iterrows():
        train_wb = get_workbook(row["file_path"])
        train_ws = train_wb[row["sheet_name"]]

        train_df = get_cell_features(train_ws)
        train_df = apply_neighbouring_features(train_df, (1, 1))

        train_df["is_header"] = 0
        ranges = str(row["header_range"]).split(";")
        for header_range in ranges:
            min_col, min_row, max_col, max_row = range_boundaries(header_range)
            header_coordinates = set([get_column_letter(i) + str(j)
                                      for i in range(min_col, max_col + 1)
                                      for j in range(min_row, max_row + 1)])
            train_df.loc[train_df["coordinate"].isin(header_coordinates), "is_header"] = 1

        train_df["file_path"] = row["file_path"]
        train_df["sheet_name"] = row["sheet_name"]

        train_wb.close()
        train_dfs.append(train_df)

    result_df = pd.concat(train_dfs).reset_index(drop=True)

    for column in result_df.columns.difference(["coordinate", "file_path", "sheet_name"]):
        result_df[column] = result_df[column].astype(bool).astype(int)

    result_df.to_pickle(export_path)

    return result_df


def balance_train_data(train_df, balance: Optional[BalanceMode] = BalanceMode.down_sample):
    result_df = train_df.copy()
    if balance == BalanceMode.down_sample:
        header_class_count = len(result_df[result_df["is_header"] == 1])
        non_header_class_count = len(result_df[result_df["is_header"] == 0])
        to_delete_non_header_count = max(0, int(0.7 * non_header_class_count - header_class_count))
        indices_to_delete = result_df[result_df["is_header"] == 0].sample(to_delete_non_header_count).index
        print("Downsampling from {} ".format(len(result_df)), end="")
        result_df = result_df.drop(indices_to_delete).reset_index(drop=True)
        print("to {} ".format(len(result_df)))
    return result_df


def get_backup_train_data(backup_path=r".\train\train.pkl"):
    return pd.read_pickle(backup_path)


def get_test_data(file_path, sheet_name="Sheet1"):
    test_wb = get_workbook(file_path)
    test_ws = test_wb[sheet_name]

    test_df = get_cell_features(test_ws)
    test_df = apply_neighbouring_features(test_df, (1, 1))

    for column in test_df.columns.difference(["coordinate"]):
        test_df[column] = test_df[column].astype(bool).astype(int)

    return test_df
