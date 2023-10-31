import pandas as pd
import openpyxl


def get_cell_features(cur_cell):
    cell_features = {
        "coordinate": cur_cell.coordinate,
        "is_empty": cur_cell.value is None,
        "is_string": cur_cell.data_type in ['s', 'str'],
        "is_merged": type(cur_cell).__name__ == 'MergedCell',
        "is_bold": cur_cell.font.b,
        "is_italic": cur_cell.font.i,
        "left_border": cur_cell.border.left is not None,
        "right_border": cur_cell.border.right is not None,
        "top_border": cur_cell.border.top is not None,
        "bottom_border": cur_cell.border.bottom is not None,
        "is_filled": cur_cell.fill.patternType is not None,
        "horizontal_alignment": cur_cell.alignment.horizontal is not None,
        "left_horizontal_alignment": cur_cell.alignment.horizontal == 'left',
        "right_horizontal_alignment": cur_cell.alignment.horizontal == 'right',
        "center_horizontal_alignment": cur_cell.alignment.horizontal == 'center',
        "wrapped_text": cur_cell.alignment.wrapText,
        "indent": cur_cell.alignment.indent != 0,
        "formula": cur_cell.data_type == 'f'
    }

    return cell_features


def get_table_features(file_path, sheet_name) -> pd.DataFrame:
    wb = openpyxl.load_workbook(file_path, read_only=True)
    ws = wb[sheet_name]
    data = []

    for row in ws.iter_rows():
        for cell in row:
            data.append(get_cell_features(cell))

    result_df = pd.DataFrame(data)

    return result_df
