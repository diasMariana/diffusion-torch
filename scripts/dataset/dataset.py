import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, data_dir, extensions=["png", "jpg"]):

        # Find data
        self.dataset = []
        print("Looking for data...")
        for root, _, files in tqdm(os.walk(data_dir)):
            for fname in files:
                extension = (fname.split(os.sep)[-1]).split(".")[-1]
                if extension in extensions:
                    self.dataset.append(os.path.join(root, fname))
        print("Done!")

        # Data transforms - to tensor and normalize to [-1,1] range
        self.normalized_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im = Image.open(self.dataset[index])
        im_tensor = self.normalized_tensor(im)
        return im_tensor
