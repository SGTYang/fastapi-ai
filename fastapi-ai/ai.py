import torch, copy, os
from torch import optim
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn
from torch.optim import lr_scheduler
from PIL import Image
from aiflow import Mlflow
from torchmetrics import F1Score
from sklearn.metrics import f1_score

MLFLOW_HOST = os.environ.get("MLFLOW_URL", "http://127.0.0.1:5000")

class GenerateDataset(Dataset):
    def __init__(self, image: list, label: list, transform = None):
        self.image, self.label = image, label
        self.transform = transform
    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):
        return self.transform(Image.open(self.image[index])), self.label[index]

class ModelTrain():
    def __init__(self, option):
        self.option = option
        
        self.device = torch.device("mps") if torch.cuda.is_available() else torch.device("cpu")
        
        self.model = EfficientNet.from_pretrained(option.model_name, num_classes=len(option.query_match_items))
        
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr = 0.05,
            momentum = 0.9,
            weight_decay = 1e-4
        )
        self.model.to(self.device)
    
    def getF1score(self, preds, target):
        num_classes = len(self.option.query_match_items)
        goal = "multiclass" if num_classes > 1 else "binary"
        f1 = F1Score(task = goal, num_classes=num_classes)
        return f1(preds, target)
    
    def learningRateScheduler(self, step_size:int, gamma:float) -> lr_scheduler:
        return lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
    
    def tensor(self, dataset: dict) -> GenerateDataset:
        image_size = EfficientNet.get_image_size(self.option.model_name)
        
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size)]
            )
        
        image_path, image_label = map(list, zip(*dataset.copy().items()))
        
        return GenerateDataset(image_path, image_label, transform=transform)
    
    def dataload(self, data: Dataset) -> DataLoader:
        return DataLoader(
            dataset = data, 
            batch_size = self.option.batch_size, 
            shuffle = self.option.shuffle
            )

    def splitData(self, total_size: int) -> list:
        train_size = round((total_size)*self.option.train_dataset_ratio/10)
        cross_val_size = (total_size-train_size)//2
        test_size = total_size - train_size - cross_val_size
        
        return [
            [train_size, cross_val_size, test_size],
            [(0,train_size), (train_size, cross_val_size+train_size), (cross_val_size+train_size, cross_val_size+train_size+test_size)]
            ]
    
    def loops(self, loaded_dataset: dict, stage_dataset_size: dict) -> tuple:
        best_acc = float('-inf')
        idx = 0
        
        learning_rate_scheduler = self.learningRateScheduler(7,0.1)
        
        loss_algo = nn.CrossEntropyLoss()

        for level in [["train", "cross_validation"], ["test"]]:
            while idx < self.option.epoch_size:
                for stage in level:
                    
                    match stage:
                        case "train":
                            self.model.train()
                        case "cross_validation":
                            self.model.eval()
                        case "test":
                            self.model.eval()
                            self.model.load_state_dict(best_model_weight)
                    
                    running_corrects = running_loss = f1 = 0.                    
                    
                    for input, label in loaded_dataset[stage]:
                        input, label = input.to(self.device), label.to(self.device)
                        
                        if stage != "test": self.optimizer.zero_grad()
                        
                        with torch.set_grad_enabled(stage == "train"):
                            output = self.model(input)
                            _, prediction = torch.max(output, 1)
                            loss = loss_algo(output, label)
                            
                            if stage == "train":
                                loss.backward()
                                self.optimizer.step()

                        running_loss += loss.item() * input.size(0)
                        running_corrects += torch.sum(prediction == label.data)
                        
                        f1 += f1_score(label.cpu().data, prediction.cpu(), average="micro")
                        
                    if stage == "train":
                        learning_rate_scheduler.step()
                    
                    epoch_loss = running_loss / stage_dataset_size[stage]
                    epoch_acc = running_corrects.double() / stage_dataset_size[stage]
                    f1score = f1/len(loaded_dataset[stage])
                    
                    if stage == 'cross_validation' and epoch_acc > best_acc:
                        best_acc = max(best_acc, epoch_acc)
                        best_model_weight = copy.deepcopy(self.model.state_dict())

                    yield [
                        {
                            "stage": f"{stage}",
                            "epoch": idx,
                            "loss": epoch_loss,
                            "accuracy": epoch_acc,
                            "f1_score": f1score
                        },
                        {"model_state_dict": self.model.state_dict()}
                    ]
                idx += 1
            self.option.epoch_size += 1 

    def airun(self, dir_class_list: list, num_dir_min_len: int) -> dict:
        mlflow_object = Mlflow(self.option.experiment_name, MLFLOW_HOST)
        
        if not mlflow_object: return {"Error": "MLflow server is not reachable"}
        
        if len(dir_class_list) != len(self.option.query_match_items):
            return {"ERROR": "Not enough classes"}
        
        image_label_dict = {val:idx for idx,val in enumerate(self.option.query_match_items)}
        
        _, split_range = self.splitData(num_dir_min_len)
        stage_and_size = list(zip(stages:=["train", "cross_validation", "test"], split_range))
        
        splited_dataset = {
            stage: self.tensor(
                {image_dir: image_label_dict[label] for class_elem in dir_class_list for image_dir, label in class_elem[start:end]}
                ) for stage,(start,end) in stage_and_size
            }
        
        stage_dataset_size = {stage: len(splited_dataset[stage])for stage in stages}

        loaded_dataset = {stage: self.dataload(splited_dataset[stage]) for stage in stages}
        
        mlflow_object.start()
        mlflow_object.settag(self.option.tag)
        mlflow_object.logparams(
            {
                "query_match_items": self.option.query_match_items,
                "epoch_size": self.option.epoch_size,
                "batch_size": self.option.batch_size,
                "train_dataset_ratio": self.option.train_dataset_ratio,
                "dataset_size": stage_dataset_size
                }
            )

        for i in self.loops(loaded_dataset, stage_dataset_size):
            mlflow_object.logmetric(i[0])
            mlflow_object.logModelDict(i[1]["model_state_dict"], self.option.model_name)

        mlflow_object.logModel(self.model, self.option.model_name)

        mlflow_object.end()
        
        return {
            "splited_data_size": stage_dataset_size
        }