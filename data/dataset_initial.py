import os
import subprocess

Path_4_Downloads = "raw/"
Path_4_Datasets = "library/"
datasets = []


class dataset:
    def __init__(self, _name, _filetype):
        self.uid = len(datasets)
        self.name = _name

        self.filetype = _filetype

        self.raw_path = f"{Path_4_Downloads}{_name}.{self.filetype}"
        self.library_path = f"{Path_4_Datasets}"

        print(self)

        datasets.append(self)
        print(f"Writen into datasets list\nDatasets[{self.uid}]:\n{datasets[self.uid]}")

    def __str__(self):
        return (
            f"Name: {self.name}\n"
            f" - Raw Path: {self.raw_path}\n"
            f" - Library Path: {self.library_path}\n"
        )


def extract_file(zip_path, extract_to):
    """Extract packages using tar with progress output."""
    if not os.path.isfile(zip_path):
        print(f"File {zip_path} does not exist.")
        exit(1)

    os.makedirs(extract_to, exist_ok=True)

    try:
        result = subprocess.run(
            ['unzip', '-o', zip_path, '-d', extract_to],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            print(f"File {zip_path} has been extracted into {extract_to}.")
        else:
            print(f"Error extracting file {zip_path}: {result.stderr}")
            exit(1)

    except FileNotFoundError:
        print("unzip command not found. Please install unzip and try again.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)


def clear_path(target_path):
    """Clear all files and directories in the target path."""
    if not os.path.exists(target_path):
        print(f"Path {target_path} does not exist.")
        return

    try:
        for root, dirs, files in os.walk(target_path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        print(f"Cleared all files and directories in {target_path}.")
    except Exception as e:
        print(f"Failed to clear path {target_path}: {e}")

if __name__ == "__main__":
    dataset("LV-MHP-v2", "zip")
    clear_path("./library/")

    for each in datasets:
        print(f"Extracting {each.name}")
        extract_file(each.raw_path, each.library_path)