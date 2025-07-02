import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
#--------------------------------------------------------------------------

class RetinalDataset(Dataset):
    def __init__(self, folder_path, target_size=(512, 512), transform=None):
        """
        Args:
            folder_path (str): Path to directory containing retinal images
            target_size (tuple): Output image dimensions (h, w)
            transform (callable, optional): Additional torchvision transforms
        """
        self.folder_path = folder_path
        self.target_size = target_size
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ]

        # Core preprocessing 
        self.base_preprocess = transforms.Compose([
            transforms.Lambda(lambda x: self._load_and_preprocess(x)),
            transforms.ToTensor()
        ])

    def _load_and_preprocess(self, image_path):
        # 1. Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image at {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Resize
        img = cv2.resize(img, (self.target_size[1], self.target_size[0]))

        # 3. CLAHE per channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channels = [clahe.apply(img[:, :, i]) for i in range(3)]
        enhanced = np.stack(channels, axis=-1).astype(np.float32) / 255.0
        return enhanced

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])

        try:
            # Apply base preprocessing
            img_tensor = self.base_preprocess(image_path)

            # Apply additional transforms if specified
            if self.transform:
                img_tensor = self.transform(img_tensor)

            return img_tensor

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            # Return empty tensor if processing fails
            return torch.zeros(3, *self.target_size)

    def get_raw_and_processed(self, idx):
        """Return original and preprocessed image for visualization"""
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        
        # Load original with OpenCV
        raw_img = cv2.imread(image_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        #raw_img_resized = cv2.resize(raw_img, (self.target_size[1], self.target_size[0]))

        # Preprocessed
        processed_img = self._load_and_preprocess(image_path)

        return raw_img, processed_img