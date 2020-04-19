import os
from PIL import Image
from torch.utils.data import Dataset


class LoadData(Dataset):
    def __init__(self, root='', transform=None):
        super(LoadData, self).__init__()
        self.files = []
        for img in os.listdir(root):
            self.files.append(os.path.join(root, img))
        self.transform = transform

    def __getitem__(self, item):
        img = Image.open(self.files[item])
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)
