import os
import pandas as pd
import subprocess
from tqdm import tqdm
from .dataset_loader import DatasetLoader
from .markup_loader import MarkupLoader
from .cell_features import get_table_features


def convert_file(file_path, output_dir):
    output_file_path = os.path.splitext(file_path)[0] + '.xlsx'
    if os.path.exists(output_file_path):
        return  # Skip if .xlsx file already exists
    try:
        subprocess.run([
            'libreoffice', '--headless', '--convert-to',
            'xlsx', file_path,
            '--outdir', output_dir
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {file_path}: {e}")


def convert_files(files_df, data_folder_path):
    # Create a copy of the original DataFrame to preserve other columns
    updated_files_df = files_df.copy()

    for index, row in tqdm(files_df.iterrows(), total=files_df.shape[0], desc="Converting Files"):
        data_file_path = os.path.join(row['parent_path'], row['file_name'])
        file_path = os.path.join(data_folder_path, data_file_path)
        output_directory = os.path.join(data_folder_path, row['parent_path'])
        file_name, file_ext = os.path.splitext(row['file_name'])

        if file_ext.lower() in ['.xls', '.xlsb']:
            convert_file(file_path, output_directory)
            # Update only the file name's extension to .xlsx
            updated_files_df.at[index, 'file_name'] = file_name + '.xlsx'

    return updated_files_df


def extract_features(files_df, data_folder_path):
    features_dfs = []
    # Iterate over the unique pairs
    for _, row in tqdm(files_df.iterrows(), total=files_df.shape[0], desc="Extracting features from files"):
        file_path = os.path.join(row['parent_path'], row['file_name'])
        features_df = get_table_features(os.path.join(data_folder_path, file_path), row['sheet_name'])
        features_df["file_path"] = file_path
        features_df["sheet_name"] = row['sheet_name']
        features_df["set_type"] = row["set_type"]
        features_df["table_range"] = [row["table_range"] for _ in range(len(features_df.index))]
        features_dfs.append(features_df)
    return pd.concat(features_dfs, ignore_index=True)


def get_train_test(train_size=30, testing_size=10, 
                   data_folder_path="./ExcelTableCNN/data/", 
                   dataset_name="VEnron2", markup_name="tablesense",
                   backup_result=True):
    print("Downloading dataset...")
    dataset_loader = DatasetLoader(save_path=data_folder_path)
    dataset_loader.get_dataset(dataset_name)
    dataset_files = dataset_loader.get_files(dataset_name)

    print("Getting markup...")
    markup_loader = MarkupLoader()
    markup_files = markup_loader.get_markup(markup_name)

    # different extensions fix:
    dataset_files["file_name_no_ext"] = dataset_files["file_name"].apply(lambda x: os.path.splitext(x)[0])
    markup_files["file_name_no_ext"] = markup_files["file_name"].apply(lambda x: os.path.splitext(x)[0])
    
    files_df = markup_files.merge(dataset_files, how="inner", on=["file_name_no_ext", "parent_path"])
    files_df = files_df.drop(columns=["file_name_x", "file_name_no_ext"])
    files_df = files_df.rename(columns={"file_name_y": "file_name"})

    training_samples = files_df[files_df["set_type"] == "training_set"].sample(train_size)
    testing_samples = files_df[files_df["set_type"] == "testing_set"].sample(testing_size)
    files_df_sample = pd.concat([training_samples, testing_samples])

    # Converting files
    dataset_files_converted = convert_files(files_df_sample, data_folder_path)

    # Getting table features
    features_df = extract_features(dataset_files_converted, data_folder_path)

    train_df = features_df[features_df["set_type"] == "training_set"].drop(columns=["set_type"])
    test_df = features_df[features_df["set_type"] == "testing_set"].drop(columns=["set_type"])

    if backup_result:
        print("Backing up results...")
        train_df.to_pickle(os.path.join(data_folder_path, "train_features.pkl"))
        test_df.to_pickle(os.path.join(data_folder_path, "test_features.pkl"))

    return train_df, test_df
