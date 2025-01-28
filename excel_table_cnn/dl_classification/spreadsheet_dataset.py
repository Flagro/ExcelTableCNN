import torch
from torch.utils.data import Dataset

from .tensors import DataframeTensors


def get_bounding_box(table_ranges):
    boxes = torch.tensor(
        [
            [x_min, y_min, x_max, y_max]
            for (x_min, y_min), (x_max, y_max) in table_ranges
        ],
        dtype=torch.float32,
    )
    # Assuming '1' is the label for tables:
    labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
    return {"boxes": boxes, "labels": labels}


class SpreadsheetDataset(Dataset):
    def __init__(self, tensors: DataframeTensors):
        self.tensors = tensors

    def get_num_cell_features(self):
        return self.data[0].shape[0]

    def __len__(self):
        # The length of the dataset is the number of spreadsheets
        return len(self.tensors.hwc_tensors)

    def __getitem__(self, idx):
        tensor = self.tensors.hwc_tensors[idx]
        # Permute tensor to C x H x W
        tensor = tensor.permute(2, 0, 1)

        # Get labels
        labels = get_bounding_box(self.tensors.zero_indexed_table_ranges[idx])

        return tensor, labels
