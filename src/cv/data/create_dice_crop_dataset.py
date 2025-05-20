import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

root_path = os.path.join("data")


def create_label_csv(label_studio_csv):
    # create label pandas dataframe
    # Read the CSV file
    df = pd.read_csv(label_studio_csv)
    # Collect entries
    data = []
    for index, row in df.iterrows():
        row_dict = row.to_dict()
        # Extract prefix and index from filename
        filename = row_dict["image"].split("/")[-1]
        label_str = row_dict["choice"]
        if isinstance(label_str, float) and  np.isnan(label_str):
            print(f"Label is NaN for {filename}. Skipping.")
            continue
        if label_str == "worm":
            label = 6
        elif label_str == "undetectable":
            continue  # skip undetectable
        else:
            label = int(label_str)
        # add to data
        data.append({
            "filename": filename,
            "label": label,
        })

    # Create DataFrame and save
    df = pd.DataFrame(data)

    # Split into train (80%) and test (20%)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    # save to csv
    train_df.to_csv(os.path.join(root_path, "dice_crop_dataset", "train.csv"), index=False)
    test_df.to_csv(os.path.join(root_path, "dice_crop_dataset", "test.csv"), index=False)

def resize_and_convert_images(input_folder, output_folder="data/dice_crop_dataset/images"):
    for idx, file in enumerate(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, file)
        output_file_path = os.path.join(output_folder, file)
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
    create_label_csv("data/exported_labels/crops.csv")
    # input_folder = "data/dice_crops"
    # output_folder = "data/dice_crop_dataset/images"
    # resize_and_convert_images(input_folder,  output_folder=output_folder)
