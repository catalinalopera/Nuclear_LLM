import zipfile
import os

class ZipExtractor:
    def __init__(self, zip_path, extract_to):
        self.zip_path = zip_path
        self.extract_to = extract_to

    def extract(self):
        # Check if the directory exists, if not, create it
        if not os.path.exists(self.extract_to):
            os.makedirs(self.extract_to)
        
        # Extract the zip file
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_to)

        print(f"Extracted {self.zip_path} to {self.extract_to}")
