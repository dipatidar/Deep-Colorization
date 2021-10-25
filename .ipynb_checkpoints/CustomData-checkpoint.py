from torch.utils.data import Dataset
import os
import random
import cv2
import numpy as np
import torch

class ImageDataset(Dataset):
    """Colrization image dataset."""

    def __init__(self, root_dir, transform, regressor_only):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        image_paths = os.listdir(root_dir)
        image_paths = [image_path for image_path in image_paths if image_path.endswith('.jpg')]
        random.shuffle(image_paths)
        
        self.image_paths = np.array(image_paths)
        self.root_dir = root_dir
        self.transform = transform
        self.regressor_only = regressor_only

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (None, None, None)
        img_name = os.path.join(self.root_dir,
                                self.image_paths[idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        if self.transform:
          # Different data depending on if regressor
          # or colorization network
          if self.regressor_only:
            image_a = image_lab[1:2, :, :]
            image_b = image_lab[2:3, :, :]
            sample = (image_gray, image_a, image_b)
            
          else:
            image_lab = self.transform(image)
            image_gray = image_lab[0:1,:,:]
            image_ab = self.transform(image)[1:,:,:]
            
            sample = (image_gray, image_lab, image_ab)

        return sample