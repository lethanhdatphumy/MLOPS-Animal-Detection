from pathlib import Path
import os
import urllib.request as request
import zipfile

from src.cnnClassifier.entity.config_entity import DataIngestionConfig
from src.cnnClassifier.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """Store ingestion config (source URL, local paths)."""
        self.config = config

    def download_data(self) -> None:
        """Download the dataset zip from the configured URL, skipping if already present."""
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            print(f"File downloaded: {filename} with info: {headers}")
        else:
            print(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self) -> None:
        """Extract the downloaded zip into the configured unzip directory."""
        unzip_path = self.config.unzip_dir
        zip_file_path = self.config.local_data_file

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)