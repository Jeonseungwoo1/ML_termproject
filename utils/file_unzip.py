import zipfile
import os

def unzip_file(zip_path, extract_to):
    """
    Unzips a zip file to the specified directory.
    
    Parameters:
    zip_path (str): The path to the zip file.
    extract_to (str): The directory to extract the contents to.
    """

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f'Files have been extracted to {extract_to}')

if __name__ == '__main__':
    # Example usage: Modify the paths as needed
    zip_path = './datasets.zip'  # Path to the zip file
    extract_to = './datasets'  # Target folder for extraction

    unzip_file(zip_path, extract_to)