import torch, copy, time, os, datetime
from torch import optim
from torch.utils.data import DataLoader, random_split
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn
from torch.optim import lr_scheduler
from PIL import Image

'''tensor 오브젝트로 바꾸기 위한 dataset생성 class'''
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

def makeTensor(dataset: dict, image_size: int):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(image_size),
         transforms.CenterCrop(image_size)]
        )
    image_path, image_label = map(list, zip(*dataset.copy().items()))
    return GenerateDataset(image_path, image_label, transform=transform)

def trainMain(option, dataset: dict):
    start = time.time()
    weight_list = []
    loss_algo = nn.CrossEntropyLoss()
    
    model_input_size, model = setModel(option.model_name, len(option.query_match_items))
    
    # test 데이터가 학습 데이터와 완전히 분리되고난 이후 모델 저장 및 불러오기
    '''
    if os.path.isfile("./model_state_dict.pth"):
        old_state = torch.load("./model_state_dict.pth")
        weight_list.append(old_state)
        model.load_state_dict(old_state)
    '''
    
    tensor_dataset = makeTensor(dataset, model_input_size)
    model.to((device := setDevice()))
    
    optimizer_ft = optim.SGD(
        model.parameters(),
        lr = 0.05,
        momentum=0.9,
        weight_decay=1e-4
        )

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    '''train, cross val, test 데이터로 나누기'''
    train_size = round((dataset_size := len(dataset))*option.train_dataset_ratio/10)
    cross_val_size = round(dataset_size*(1-option.train_dataset_ratio/10)/2)
    test_size = dataset_size - train_size - cross_val_size
    
    '''나누어진 데이터셋을 dict형식으로 변환'''
    splited_dataset = dict(
        zip(
            (stages := ["train", "cross_validation", "test"]),
            random_split(tensor_dataset, [train_size, cross_val_size, test_size])
            )
        )
    
    '''dataloader로 데이터셋 로드'''
    loaded_dataset = {
        stage: 
            DataLoader(
                dataset = splited_dataset[stage],
                batch_size = option.batch_size,
                shuffle = option.shuffle
                ) 
            for stage in stages
        }
    
    (stage_dataset_size := {stage: len(splited_dataset[stage]) for stage in stages})

    cross_validation_best_acc = 0.
    total_epoch = 0
    
    while total_epoch < option.epoch_size:
        total_epoch += 1
        
        for stage in ["train", "cross_validation"]:
            match stage:
                case "train":
                    model.train()
                case "cross_validation":
                    model.eval()
            
            running_loss = 0.
            running_corrects = 0
        
            for input,label in loaded_dataset[stage]:
                input, label = input.to(device), label.to(device)
                
                optimizer_ft.zero_grad()
                
                with torch.set_grad_enabled(stage=="train"):
                    output = model(input)
                    _, prediction = torch.max(output, 1)
                    loss = loss_algo(output, label)
                
                    if stage == "train":
                        loss.backward()
                        optimizer_ft.step()
                        
                running_loss += loss.item() * input.size(0)
                running_corrects += torch.sum(prediction == label.data)

            if stage == "train":
                exp_lr_scheduler.step()
                    
            epoch_loss = running_loss / stage_dataset_size[stage]
            epoch_acc = running_corrects.double() / stage_dataset_size[stage]

            print(f'{stage} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if stage == 'cross_validation' and epoch_acc > cross_validation_best_acc:
                cross_validation_best_acc = epoch_acc
                best_model_weight = copy.deepcopy(model.state_dict())
    
    weight_list.append(best_model_weight)
    
    test_max_acc = float("-inf")

    '''
    현재: 훈련 전후 비교해 더 좋은 weight를 가져오게 되어있음
    
    확장: 여러 다른 모델을 비교해 더 좋은 모델과 weight를 저장 
        -> 여러 모델 필요 및 모델 학습시 multithread 방식으로 여러 모델 동시 학습(thread별 GPU 접근 방법 조사 필요)
    '''
    for weight in weight_list:
        model.load_state_dict(weight)
        model.eval()
        test_corrects = 0
    
        for input,label in loaded_dataset["test"]:
            input, label = input.to(device), label.to(device)
            
            output = model(input)
            _, prediction = torch.max(output, 1)
            loss = loss_algo(output, label)
            test_corrects += torch.sum(prediction == label.data)

        total_acc = test_corrects.double() / stage_dataset_size["test"]

        if test_max_acc < total_acc:
            test_max_acc = total_acc
            best_wieght = copy.deepcopy(model.state_dict())
    
    time_elapsed = time.time() - start
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best cross_validation_best_acc:{cross_validation_best_acc:4f}")
    
    model.load_state_dict(best_wieght)
    
    current_time = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    
    '''
    로컬에 물리 모델 저장 네이밍 정해지면 /data에 저장 되도록 수정 및 ai service에서도 /data에 접근해 
    배포된 모델 사용하도록 기능 생성
    '''
    torch.save(model, f"/data/model/{option.model_name}-{current_time}.pth")
    # inference용
    torch.save(model.state_dict(), f"/data/model/{option.model_name}-{current_time}_state_dict.pth")
    
    '''front로 return'''
    return {
        "total_epochs": f"{total_epoch}",
        "bach_size": f"{option.batch_size}",
        "cross_validation_best_acc": f"{cross_validation_best_acc:4f}",
        "training_time": f"{time_elapsed//60:.0f}m {time_elapsed%60:.0f}s",
        "test_max_acc": f"{test_max_acc:.4f}",
        "splited_data_size": stage_dataset_size,
        "model": model,
        }