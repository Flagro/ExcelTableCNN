import os
import requests
from py7zr import unpack_7zarchive
import shutil

class DatasetLoader:
    def __init__(self, save_path):
        # Where the datasets will be saved
        self.save_path = save_path
        
        # URLs of datasets
        self.datasets = {
            'VEnron2': 'https://figshare.com/ndownloader/files/8639470',
            'VFUSE': 'https://figshare.com/ndownloader/files/7889911',
            'VEUSES': 'https://figshare.com/ndownloader/files/7889902',
            'VEnron1.1': 'https://figshare.com/ndownloader/files/7889866',
            'VEnron1.0': 'https://figshare.com/ndownloader/files/7889947'
        }

    def download_dataset(self, dataset_name):
        # Check if dataset_name is valid
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found.")
            return
        
        url = self.datasets[dataset_name]
        response = requests.get(url, stream=True)
        dataset_path = os.path.join(self.save_path, f"{dataset_name}.7z")
        
        with open(dataset_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Dataset {dataset_name} downloaded successfully.")
        
    def unpack_dataset(self, dataset_name):
        dataset_path = os.path.join(self.save_path, f"{dataset_name}.7z")
        extract_path = os.path.join(self.save_path, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_name} not found at {dataset_path}.")
            return
        
        # Register the .7z format with shutil
        shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)
        shutil.unpack_archive(dataset_path, extract_path, '7zip')
        print(f"Dataset {dataset_name} unpacked successfully.")
        
    def get_dataset(self, dataset_name):
        self.download_dataset(dataset_name)
        self.unpack_dataset(dataset_name)
