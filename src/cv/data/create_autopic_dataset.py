import os

from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

root_path = os.path.join("data")


def create_label_csv():
    # create label pandas dataframe

    # Collect entries
    data = []
    for file in os.listdir(os.path.join(root_path, "autopic_dataset", "images")):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            # Extract prefix and index from filename
            prefix, index = file.split("_", 1)
            if prefix == "accepted":
                label = 1
            else:
                label = 0
            # add to data
            data.append({
                "filename": file,
                "autopic_usable": label,
                "prefix": prefix,
            })

    # Create DataFrame and save
    df = pd.DataFrame(data)

    # Split into train (80%) and test (20%)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df["autopic_usable"])
    # save to csv
    train_df.to_csv(os.path.join(root_path, "autopic_dataset", "train.csv"), index=False)
    test_df.to_csv(os.path.join(root_path, "autopic_dataset", "test.csv"), index=False)

def resize_and_convert_images(input_folder, prefix, output_folder="data/autopic_dataset/images"):
    # Walk through the folder and subfolders
    for idx, file in enumerate(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, file)
        output_file_path = os.path.join(output_folder, f"{prefix}_{idx}.jpg")
        # check if exists
        if os.path.exists(output_file_path):
            print(f"File already exists: {output_file_path}")
            continue
        # Check if the file is a valid image
        if file.endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Open and resize image
                with Image.open(file_path) as img:
                    img_resized = img.resize((256, 256))
                    img_rgb = img_resized.convert("RGB")  # Ensure compatible for JPEG
                    # Save as JPEG
                    img_rgb.save(output_file_path, format="JPEG")
                    print(f"Saved resized image: {output_file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    create_label_csv()

    # input_folders = ["data/accepted_frames",
    #                 "data/rejected_frames",
    #                 "data/negative_samples"]
    # prefixes = ["accepted", "rejected", "negative"]
    # output_folder = "data/autopic_dataset/images"
    # for input_folder, prefix in zip(input_folders, prefixes):
    #     # Resize and convert images
    #     resize_and_convert_images(input_folder, prefix, output_folder=output_folder)


