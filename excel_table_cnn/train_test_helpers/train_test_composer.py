import os
import pandas as pd
import subprocess
from tqdm import tqdm
from train_test_helpers import DatasetLoader, MarkupLoader


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
    new_rows = []
    for _, row in tqdm(files_df.iterrows(), total=files_df.shape[0], desc="Converting Files"):
        data_file_path = os.path.join(row['parent_path'], row['file_name'])
        file_path = os.path.join(data_folder_path, data_file_path)
        output_directory = os.path.join(data_folder_path, row['parent_path'])
        file_name, file_ext = os.path.splitext(row['file_name'])

        if file_ext.lower() in ['.xls', '.xlsb']:
            convert_file(file_path, output_directory)
            new_rows.append({'file_name': file_name + '.xlsx', 'parent_path': row['parent_path']})

    new_files_df = pd.DataFrame(new_rows)
    return new_files_df


def get_train_test(train_size=30, testing_size=10, 
                   data_folder_path="./ExcelTableCNN/data/", 
                   dataset_name="VEnron2", markup_name="tablesense"):
    dataset_loader = DatasetLoader(save_path=data_folder_path)
    dataset_loader.get_dataset(dataset_name)
    dataset_files = dataset_loader.get_files(dataset_name)

    markup_loader = MarkupLoader()
    markup_files = markup_loader.get_markup(markup_name)
    
    files_df = markup_files.merge(dataset_files, how="inner", on=["file_name_no_ext", "parent_path"])
    files_df = files_df.drop(columns=["file_name_x", "file_name_no_ext"])
    files_df = files_df.rename(columns={"file_name_y": "file_name"})

    training_samples = files_df[files_df["set_type"] == "training_set"].sample(train_size)
    testing_samples = files_df[files_df["set_type"] == "testing_set"].sample(testing_size)
    files_df_sample = pd.concat([training_samples, testing_samples])

    # dataset_files_converted = convert_files(files_df_sample, data_folder_path)

    return files_df_sample
