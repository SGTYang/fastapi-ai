from tensor import getTensort

import torch
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

num_classes = 12
model_name = 'efficientnet-b0'
image_size = EfficientNet.get_image_size(model_name)
model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

image_path_list = ["test1", "test2", "test3", "test4"]

class GenerateDataset(Dataset):
    def __init__(self, image_path_list: list, label: str, transform = None):
        self.image_path_list = image_path_list
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img_path = self.image_path_list[index]
        img = self.transform(Image.open(img_path))
        return img, self.label

def makeTensor(image_path_list):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize
    ])
    return GenerateDataset(image_path_list, "test", transform=transform)


def runTrain(TrainOption, image_path_list):
    
    option = TrainOption.copy()
    
    path_list = image_path_list.copy()
    
    loaded_train_dataset = DataLoader(
        
        dataset = makeTensor(path_list),
        
        batch_size = option.batch_size,
        
        shuffle = option.shuffle,
        
        sampler= option.sampler,
        
        batch_sampler = option.batch_sampler,
        
        num_workers = option.num_workers,
        
        collate_fn = option.collate_fn,
        
        pin_memory = option.pin_memory,
        
        drop_last = option.drop_last,
        
        timeout = option.timeout,
        
        worker_init_fn = option.worker_init_fn,
        
        prefetch_factor = option.prefetch_factor,
        
        persistent_workers = option.persistent_workers,
    )
    
    for _ in range(option.epoch_size):
        for input, label in loaded_train_dataset:
            pass