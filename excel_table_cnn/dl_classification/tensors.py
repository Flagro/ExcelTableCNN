import pandas as pd
import numpy as np
import re

def coordinate_to_row_col(coord):
    """Convert Excel-style coordinate to row and column index."""
    col_letters, row_num = re.match(r'([A-Z]+)([0-9]+)', coord).groups()
    col_idx = sum([(ord(char) - 64) * (26**i) for i, char in enumerate(col_letters[::-1])]) - 1
    row_idx = int(row_num) - 1
    return row_idx, col_idx

def extract_bounding_boxes(table_ranges):
    """Extract bounding boxes from table ranges."""
    bounding_boxes = []
    for table_range in table_ranges:
        start_range, end_range = table_range.split(":")
        start_row, start_col = coordinate_to_row_col(start_range)
        end_row, end_col = coordinate_to_row_col(end_range)
        bounding_boxes.append((start_row, start_col, end_row, end_col))
    return bounding_boxes

def tables_to_tensors(features_df):
    df = features_df.copy()  # Your dataframe

    results = {}
    FIXED_N, FIXED_M = 150, 50

    # Group by each sheet
    for (_, group) in df.groupby(['file_path', 'sheet_name']):
        tensors = []
        target_tensors = []  # To hold the target bounding boxes
        table_ranges = group['table_range'].iloc[0]
        C = len(group.columns) - 5 + 1
        
        bounding_boxes = extract_bounding_boxes(table_ranges)

        # Identify the starting point of data in this group
        min_row_idx = min(coordinate_to_row_col(coord)[0] for coord in group['coordinate'])
        min_col_idx = min(coordinate_to_row_col(coord)[1] for coord in group['coordinate'])

        for row_block_start in range(min_row_idx, min_row_idx + FIXED_N, FIXED_N):
            for col_block_start in range(min_col_idx, min_col_idx + FIXED_M, FIXED_M):
                tensor = np.zeros((FIXED_N, FIXED_M, C))
                
                block_group = group[group['coordinate'].apply(lambda coord: 
                                row_block_start <= coordinate_to_row_col(coord)[0] < row_block_start + FIXED_N and 
                                col_block_start <= coordinate_to_row_col(coord)[1] < col_block_start + FIXED_M)]
                
                # Extract bounding boxes that fall within this block
                block_bboxes = [(start_row - row_block_start, start_col - col_block_start, end_row - row_block_start, end_col - col_block_start)
                                for (start_row, start_col, end_row, end_col) in bounding_boxes
                                if row_block_start <= start_row < row_block_start + FIXED_N and
                                col_block_start <= start_col < col_block_start + FIXED_M]

                for _, row in block_group.iterrows():
                    row_idx, col_idx = coordinate_to_row_col(row['coordinate'])
                    features = row.drop(['file_path', 'sheet_name', 'coordinate', 'table_range', 'set_type']).values

                    tensor[row_idx - row_block_start, col_idx - col_block_start, :-1] = features

                tensors.append(tensor)
                target_tensors.append({"boxes": block_bboxes})

        results[(group["file_path"].iloc[0], group["sheet_name"].iloc[0])] = {"tensors": tensors, "targets": target_tensors, "set_type": group["set_type"].iloc[0]}
    return results
