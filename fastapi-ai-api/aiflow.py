import mlflow, os

BASE_DIR = os.environ.get("BASE_DIR", "/data/model")

class Mlflow():
    def __init__(self, experiment, MLFLOW_HOST):
        mlflow.set_tracking_uri(MLFLOW_HOST)
        mlflow.set_experiment(experiment) 
        self.experiment = mlflow.get_experiment_by_name(experiment)
        
    def getExInfo(self):
        return {
            "Experiment_id": f"{self.experiment.experiment_id}",
            "Artifact_location": f"{self.experiment.artifact_location}",
            "Tags": f"{self.experiment.tags}",
            "Lifecycle_stage": f"{self.experiment.lifecycle_stage}"
        }
        
    def settag(self, tags) -> None:
        if not tags: return
        mlflow.set_tags(tags)
        return
    
    def autolog(self) -> None:
        mlflow.pytorch.autolog()
        return
    
    def start(self) -> None:
        mlflow.start_run()
        return
    
    def end(self) -> None:
        mlflow.end_run()
        return
    
    def autolog(self) -> None:
        mlflow.pytorch.autolog()
        return
    
    def logparams(self, params:dict) -> None:
        if not params: return
        for param in params.keys():
            mlflow.log_param(param, params[param])
        return
    
    def logmetric(self, metrics:dict) -> bool:
        if not metrics: return None
        mlflow.log_metric(
            key = f"accuracy_{metrics['stage']}", 
            value = metrics["accuracy"],
            step = metrics["epoch"]
            )
        mlflow.log_metric(
            key = f"loss_{metrics['stage']}",
            value = metrics["loss"],
            step = metrics["epoch"]
        )
        return
    
    def logModelDict(self, model_state_dict, name) -> bool:
        if not model_state_dict or not name: return 
        mlflow.pytorch.log_state_dict(model_state_dict, artifact_path = name)
        return
    
    '''
    if a registered model with the name doesn’t exist, the method registers a new model and creates Version 1
    '''
    def logModel(self, model, name)-> bool:
        if not model or not name: return
        mlflow.pytorch.log_model(
            pytorch_model = model, 
            artifact_path = name,
            registered_model_name = f"{name}-registered"
            )
        return
    
    def registerModel(self, run_id, model_path, model_name) -> None:
        mlflow.register_model(f"runs:/{run_id}/{model_path}", f"{model_name}")
        return