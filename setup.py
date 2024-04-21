import os
import gdown
import zipfile
import shutil

def download_dataset(url, zip_path):
    try:
        gdown.download(url, zip_path, quiet=False)
        print("Dataset downloaded successfully!")
    except Exception as e:
        print("Error downloading dataset:", e)

def extract_dataset(zip_path, root_folder):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_folder)
        print("Dataset extracted successfully!")
    except Exception as e:
        print("Error extracting dataset:", e)

def remove_macosx_dir(root_folder):
    macosx_dir = os.path.join(root_folder, '__MACOSX')
    if os.path.exists(macosx_dir):
        shutil.rmtree(macosx_dir)
        print("__MACOSX directory removed.")

def remove_zip_file(zip_path):
    os.remove(zip_path)
    print("Zip file removed.")

def download_and_extract_dataset(url, root_folder):
    zip_path = os.path.join(root_folder, 'preprocessed_ori.zip')
    download_dataset(url, zip_path)
    extract_dataset(zip_path, root_folder)
    remove_macosx_dir(root_folder)
    remove_zip_file(zip_path)

if __name__ == "__main__":
    # URL to your dataset on Google Drive
    url = 'https://drive.google.com/uc?id=1_Rj4UD_xY9KiVp5jXvCi8m3q2VB-q2TW'
    

    # Root folder where you want to extract the dataset
    root_folder = '../'

    # Download and extract the dataset
    download_and_extract_dataset(url, root_folder)