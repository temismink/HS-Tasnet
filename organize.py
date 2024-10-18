import os
import shutil

# Mapping of keywords to target names
name_map = {
    "drum": "drums.wav",
    "percussion": "drums.wav",
    "bass": "bass.wav",
    "vox": "vocals.wav",
    "lead": "vocals.wav",
    "backing": "vocals.wav"
}

def categorize_file(file_name):
    """
    Determines the category of the file based on the name.
    Returns the appropriate new name (drums.wav, bass.wav, vocals.wav, or others.wav).
    """
    file_name_lower = file_name.lower()

    # Check if the file name contains keywords for drums, bass, or vocals
    for keyword, new_name in name_map.items():
        if keyword in file_name_lower:
            return new_name

    # If no match is found, categorize as 'others'
    return "others.wav"

def rename_files_in_directory(directory):
    """
    Renames files in the given directory based on the instrument or category.
    Files are renamed to drums.wav, bass.wav, vocals.wav, or others.wav.
    If there are multiple files in the same category, a number is appended to the filename.
    """
    file_counter = {
        "drums.wav": 0,
        "bass.wav": 0,
        "vocals.wav": 0,
        "others.wav": 0
    }

    # List all files in the directory
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        # Skip if it's not a file
        if not os.path.isfile(file_path):
            continue

        # Determine the new name for the file
        new_name = categorize_file(file_name)

        # If there's already a file with the same new name, append a number
        if file_counter[new_name] > 0:
            new_name = f"{new_name.split('.')[0]}_{file_counter[new_name] + 1}.wav"
        
        # Update the counter for this type of file
        file_counter[new_name.split('_')[0] + '.wav'] += 1

        # Create the new path for the renamed file
        new_file_path = os.path.join(directory, new_name)

        # Rename the file (or move it if needed)
        print(f"Renaming {file_name} to {new_name}")
        shutil.move(file_path, new_file_path)

def process_all_folders(root_dir):
    """
    Processes all song folders in the 'train' and 'valid' directories.
    """
    for split in ['train', 'valid']:
        split_dir = os.path.join(root_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} does not exist.")
            continue

        # Iterate over each song folder in the 'train' and 'valid' directories
        for folder_name in os.listdir(split_dir):
            folder_path = os.path.join(split_dir, folder_name)

            if os.path.isdir(folder_path):  # Make sure it's a directory
                print(f"Processing folder: {folder_path}")
                rename_files_in_directory(folder_path)

if __name__ == "__main__":
    # Specify the root directory containing 'train' and 'valid' directories
    root_directory = "/Users/samuelminkov/Desktop/Hybrid-spectogram Tasnet/dataset"  # Update this to your dataset path
    process_all_folders(root_directory)