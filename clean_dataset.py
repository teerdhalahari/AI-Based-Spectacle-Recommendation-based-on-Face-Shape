import os
from PIL import Image

def remove_corrupted_images(root_folder):
    print("Scanning for corrupted images...")
    removed_count = 0

    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # verify integrity
            except Exception:
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)
                removed_count += 1

    print(f"\nDone. Removed {removed_count} corrupted images.")

if __name__ == "__main__":
    remove_corrupted_images("FaceShape Dataset")