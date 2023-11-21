import torch
from torch.utils.data import Dataset
from openpyxl.utils.cell import coordinate_to_tuple, range_boundaries


def parse_coordinate(coordinate):
    # Convert Excel-style coordinate (e.g., 'A1') to numerical indices
    # Placeholder for actual implementation
    row_index, col_index = coordinate_to_tuple(coordinate)
    return (row_index - 1, col_index - 1)

def parse_table_range(table_range):
    # Convert table range string to numerical coordinates
    # Placeholder for actual implementation
    min_col, min_row, max_col, max_row = range_boundaries(table_range)
    return [(min_row - 1, min_col - 1), (max_row - 1, max_col - 1)]

def preprocess_features(row):
    # Convert the pandas Series directly to a tensor
    return torch.tensor(row.values, dtype=torch.float32)

class SpreadsheetDataset(Dataset):
    def __init__(self, dataframe):
        self.data = []
        self.labels = []

        non_feature_columns = ['coordinate', 'file_path', 'sheet_name', 'table_range']

        # Group by file_path and sheet_name to process each sheet separately
        grouped = dataframe.groupby(['file_path', 'sheet_name'])
        for _, group in grouped:
            max_rows, max_cols = self._get_max_dimensions(group)
            num_features = len(group.columns) - len(non_feature_columns)

            sheet_tensor = torch.zeros((max_rows, max_cols, num_features))
            label_grid = torch.zeros((max_rows, max_cols), dtype=torch.long)
            table_ranges = [parse_table_range(tr) for tr in group['table_range'].iloc[0]]

            for _, row in group.iterrows():
                row_idx, col_idx = parse_coordinate(row['coordinate'])
                cell_features = preprocess_features(row.drop(non_feature_columns))
                sheet_tensor[row_idx, col_idx, :] = cell_features

                # Determine label for the cell
                label = 0  # Default label (outside any table)
                for (start_row, start_col), (end_row, end_col) in table_ranges:
                    if start_row <= row_idx <= end_row and start_col <= col_idx <= end_col:
                        label = 1  # Cell is inside a table
                        break
                label_grid[row_idx, col_idx] = label

            self.data.append(sheet_tensor)
            self.labels.append(label_grid)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def _get_max_dimensions(self, group):
        # Compute the max row and column indices for this spreadsheet
        max_row, max_col = 0, 0
        for _, row in group.iterrows():
            row_idx, col_idx = parse_coordinate(row['coordinate'])
            max_row = max(max_row, row_idx)
            max_col = max(max_col, col_idx)
        return max_row + 1, max_col + 1  # Add 1 because indices are zero-based
