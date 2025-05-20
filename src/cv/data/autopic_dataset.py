import os

import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import torchvision.transforms.v2 as T


class AutopicDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, data_augmentation=True):
        self.manifest = manifest
        self.manifest = self.manifest.reset_index(drop=True)
        self.image_cache_path = os.path.join("data", "autopic_dataset", "images")
        self.transformation_pipeline = self.create_transformation_pipeline(data_augmentation)
        print(f"Created dataset with {len(self.manifest)} images.")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        # Get image path
        filename = self.manifest.loc[idx, 'filename']
        img_path = os.path.join(self.image_cache_path, filename)
        # Open and preprocess image
        image = torchvision.io.decode_image(img_path)
        # check that image is correctly loaded
        if image is None:
            raise UnidentifiedImageError(f"Image {img_path} could not be loaded.")
        assert len(image.shape) == 3
        assert image.shape[0] == 3
        # Prepare model inputs:
        inputs = self.transformation_pipeline(image)
        # label
        label = torch.tensor([self.manifest.loc[idx, 'autopic_usable']], dtype=torch.float32)
        return {'image': inputs, "label": label}

    def create_transformation_pipeline(self, data_augmentation) -> T.Compose:
        # Base transforms
        transform_list = [
            T.ConvertImageDtype(torch.float32),  # To float32 in [0, 1] if necessary
        ]
        if data_augmentation:
            # dataset augmentation
            transform_list.append(T.RandomHorizontalFlip(p=0.5))
            transform_list.append(
                T.RandomPerspective(distortion_scale=0.1,
                                    p=0.5))

            transform_list.append(T.ColorJitter(brightness=0.2,
                                                contrast=0.2,
                                                saturation=0.2,
                                                hue=0.1))

            transform_list.append(T.GaussianBlur(kernel_size=5,
                                                 sigma=[0.001, 0.5]))

        # Final transform
        transform = T.Compose(transform_list)
        return transform

    @staticmethod
    def get_autopic_label_map():
        return {'yes': 1, 'no': 0}

    @staticmethod
    def _show_tensor_image(tensor_img, title=None):
        import matplotlib
        matplotlib.use('TkAgg')
        # Ensure tensor is on CPU and detached
        img = tensor_img.detach().cpu()

        # If it's a batch, pick the first image
        if img.dim() == 4:
            img = img[0]

        # Convert (C, H, W) to (H, W, C)
        img = img.permute(1, 2, 0)

        # Clamp to [0,1] for display if needed
        img = img.clamp(0, 1).numpy()

        # Show with matplotlib
        plt.imshow(img)
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()
