import torch
import torch.nn.functional as F
import cv2

from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import config

class Dataset:
    def __init__(self, data, num_classes, transforms):
        self.images = data['images'].values
        self.masks = data['mask'].values
        self.num_classes = num_classes
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = cv2.imread(image)
        mask = cv2.imread(mask)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        trnsfrmd = self.transforms(image=image, mask=mask)

        image = trnsfrmd['image']
        mask = trnsfrmd['mask']
        mask = mask/10

        return {
            "images": image.to(torch.float32),
            "masks": mask.to(torch.long)
            }


def get_loaders(df):
    train_df, val_df = train_test_split(df, test_size=0.25, shuffle=True)

    train_transforms = A.Compose([
    A.Resize(*config.IMG_SIZE),
    ToTensorV2()
    ])
    val_transforms = A.Compose([
        A.Resize(*config.IMG_SIZE),
        ToTensorV2()
    ])

    train_data = Dataset(train_df, config.NUM_CLASSES, train_transforms)
    val_data = Dataset(val_df, config.NUM_CLASSES, val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.BATCH_SIZE,shuffle=True, pin_memory=config.PIN_MEMORY
        )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=config.BATCH_SIZE,pin_memory=config.PIN_MEMORY
        )

    return train_loader, val_loader