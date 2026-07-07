def apply_neighbouring_features(df, range_of_neighbors):
    """
    range_of_neighbors: tuple(height, width)
    """
    working_df = df.copy()
    working_df = working_df.set_index("coordinate")

    neighbouring_features = ["is_empty", "is_merged", "is_bold", "is_italic",
                             "left_border", "right_border", "top_border", "bottom_border",
                             "is_filled", "wrapped_text", "indent", "formula"]

    h, w = range_of_neighbors
    for index, row in working_df.iterrows():
        row = row.copy()
        r, c = coordinate_to_tuple(index)
        for i in range(-h, h+1):
            for j in range(-w, w+1):
                if i == 0 and j == 0:
                    continue
                new_r = r + i
                new_c = c + j
                if new_r > 0 and new_c > 0:
                    new_coordinate = get_column_letter(new_c) + str(new_r)
                    if new_coordinate in working_df.index:
                        for neighbouring_feature in neighbouring_features:
                            new_column_name = neighbouring_feature + str(i) + "_" + str(j)
                            neighbouring_value = working_df.loc[new_coordinate, neighbouring_feature]
                            if new_column_name not in working_df.columns:
                                working_df[new_column_name] = 0
                                working_df[new_column_name] = working_df[new_column_name].astype(int)
                            working_df.loc[index, new_column_name] = neighbouring_value

    working_df = working_df.reset_index()
    return working_df
