import torch
from torch import optim
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch import nn
from torch.optim import lr_scheduler

class GenerateDataset(Dataset):
    def __init__(self, image: list, label: list, transform = None):
        self.image, self.label = image, label
        self.transform = transform
    
    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        return self.transform(Image.open(self.image[index])), self.label[index]

def setDevice():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def setModel(model_name: str, num_classes: int):
    image_size = EfficientNet.get_image_size(model_name)
    pretrained = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    return (image_size, pretrained)

def makeTensor(image_path: list, label_list: list, image_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size)
    ])
    return GenerateDataset(image_path.copy(), label_list.copy(), transform=transform)

def trainMain(TrainOption, class_and_image: dict):
    option = TrainOption.copy()
    image_path, image_label = map(list, zip(*class_and_image.copy().items()))
    image_label_set = set(image_label)
    image_size, pretrained_model = setModel(option.model_name, len(image_label_set))
    device = setDevice()
    pretrained_model.to(device)
    
    optimizer_ft = optim.SGD(
        pretrained_model.parameters(),
        lr = 0.05,
        momentum=0.9,
        weight_decay=1e-4
        )
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    pretrained_model.train()
    
    image_label_dict={val:idx for idx,val in enumerate(image_label_set)}
    
    loaded_train_dataset = DataLoader(
        dataset = makeTensor(
            image_path, 
            list(map(image_label_dict.get, image_label)), 
            image_size
            ),
        batch_size = option.batch_size,
        shuffle = option.shuffle,
        # num_workers = option.num_workers,
        # collate_fn = option.collate_fn,
        # pin_memory = option.pin_memory,
        # drop_last = option.drop_last,
        # timeout = option.timeout,
        # worker_init_fn = option.worker_init_fn,
        # prefetch_factor = option.prefetch_factor,
        # persistent_workers = option.persistent_workers,
    )
    
    criterion = nn.CrossEntropyLoss()
    torch.cuda.empty_cache()
    for _ in range(option.epoch_size):
        running_loss = 0.
        running_corrects = 0
        for idx, (input,label) in enumerate(loaded_train_dataset):
            input = input.to(device)
            label = label.to(device)
            
            optimizer_ft.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = pretrained_model(input)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, label)
                
                loss.backward()
                optimizer_ft.step()
            
            running_loss += loss.item() * input.size(0)
            running_corrects += torch.sum(preds == label.data)
            print(f"Iteration:{idx+1}")
        exp_lr_scheduler.step()
        
        epoch_loss = running_loss / len(image_path)
        epoch_acc = running_corrects.double()/ len(image_path)
        torch.cuda.empty_cache()
        print(f"Loss: {epoch_loss:.4f} Acc:{epoch_acc:.4f}")
    
    return 0
