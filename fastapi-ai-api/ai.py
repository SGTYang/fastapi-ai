import torch, copy, os
from torch import optim
from torch.utils.data import DataLoader, random_split
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn
from torch.optim import lr_scheduler
from PIL import Image
from aiflow import Mlflow

MLFLOW_HOST = os.environ.get("MLFLOW_URL", "http://112.220.111.68:5600")

'''tensor 오브젝트로 바꾸기 위한 dataset생성 class'''
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
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = EfficientNet.from_pretrained(option.model_name, num_classes=len(option.query_match_items))
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr = 0.05,
            momentum = 0.9,
            weight_decay = 1e-4
        )
        self.model.to(self.device)
    
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
        '''
        dataloader 타입 반환
        '''
        return DataLoader(
            dataset = data, 
            batch_size = self.option.batch_size, 
            shuffle = self.option.shuffle
            )
    
    def split(self, total_size: int) -> list:
        '''
        train, cross_val, test로 데이터셋 split
        '''
        train_size = round((total_size)*self.option.train_dataset_ratio/10)
        cross_val_size = round(total_size*(1-self.option.train_dataset_ratio/10)/2)
        test_size = total_size - train_size - cross_val_size
        
        return [train_size, cross_val_size, test_size]
    
    def loops(self, loaded_dataset: dict, stage_dataset_size: dict) -> tuple:
        '''
        현재: 훈련전 모델과 훈련후 모델을 test 단계에서 비교후 더 좋은 모델 저장
    
        확장: 여러 다른 모델을 비교해 더 좋은 모델과 weight를 저장 
            -> 여러 모델 필요 및 모델 학습시 multithread 방식으로 여러 모델 동시 학습(thread별 GPU 접근 방법 조사 필요)
        로컬에 물리 모델 저장 네이밍 정해지면 /data에 저장 되도록 수정 및 ai service에서도 /data에 접근해 
        배포된 모델 사용하도록 기능 생성
        '''
        
        best_acc = best_loss = float('-inf')
        idx = 0
        
        ''' 
        learning rate scheduler 설정
        '''
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
                            
                    running_loss = 0.
                    running_corrects = 0.
                    
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
                        
                    if stage == "train":
                        learning_rate_scheduler.step()
                    
                    epoch_loss = running_loss / stage_dataset_size[stage]
                    epoch_acc = running_corrects.double() / stage_dataset_size[stage]

                    if stage == 'cross_validation' and epoch_acc > best_acc:
                        best_acc = max(best_acc, epoch_acc)
                        best_model_weight = copy.deepcopy(self.model.state_dict())

                    yield [
                        {
                            "stage": f"{stage}",
                            "epoch": idx,
                            "loss": epoch_loss,
                            "accuracy": epoch_acc,
                        },
                        {"model_state_dict": self.model.state_dict()}
                    ]
                idx += 1
            self.option.epoch_size += 1 

    def airun(self, dataset: dict) -> dict:
        #mlflow_object = Mlflow(self.option.experiment_name, MLFLOW_HOST) if os.system(f"ping -n 1 {MLFLOW_HOST}") == 0 else None
        mlflow_object = Mlflow(self.option.experiment_name, MLFLOW_HOST)
        
        if not mlflow_object: return {"Error": "MLflow server is not reachable"}
        
        '''self.dataset을 tensor로 변경'''
        tensor_dataset = self.tensor(dataset)
        
        '''나누어진 데이터셋을 dict형식으로 변환'''
        splited_dataset = dict(
            zip(
                (stages := ["train", "cross_validation", "test"]), 
                random_split(tensor_dataset, self.split(len(dataset)))
                )
            )
        
        '''
        loaded_dataset =
        {
            "train": DataLoader,
            "cross_validation": DataLoader,
            "test": DataLoader,
        }
        '''
        loaded_dataset = {stage: self.dataload(splited_dataset[stage]) for stage in stages}
        
        '''
        stage_dataset_size =
        {
            "train": train_size,
            "cross_validation": cross_validation_size,
            "test": test_size
        }
        '''
        
        stage_dataset_size = {stage: len(splited_dataset[stage]) for stage in stages}
        
        '''
        iterate 하면서 단계별로 train, cross_validation, test 실행
        '''
        mlflow_object.start()
        mlflow_object.autolog()
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
        
        '''
        실험후 모델선정은 수동으로 
        '''
        
        mlflow_object.logModel(self.model, self.option.model_name)
        print(mlflow_object.getExInfo())
        
        mlflow_object.fetchModel(f"{self.option.model_name}-registered", 1)

        mlflow_object.end()
        
        '''
        model_uri = "runs:/18d1d16681e04d80a39c0d1d388f365e/efficientnet-b0",
        dst_path = "./"
        '''
        
        return {
            "splited_data_size": stage_dataset_size
        }