from os.path import join
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize


def load_data(batch_size=32, source='data/celeb/celeb'):
    train_data = datasets.ImageFolder(root=join(source, 'train'),
                                      transform=Compose([ToTensor(),
                                                         Normalize(mean=[0.5], std=[0.5])
                                                         ])
                                      )

    test_data = datasets.ImageFolder(root=join(source, 'test'),
                                     transform=Compose([ToTensor(),
                                                        Normalize(mean=[0.5], std=[0.5])
                                                        ])
                                     )

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader
