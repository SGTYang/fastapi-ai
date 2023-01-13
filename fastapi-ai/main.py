from ai import ModelTrain
from fastapi import FastAPI
import os
from typing import List
from pydantic import BaseModel
from elastic import Elastic

app = FastAPI()

class Option(BaseModel):
    model_name: str = "efficientnet-b0"
    experiment_name: str = "heart"
    epoch_size: int = 2
    batch_size: int = 128
    tag: dict = {"purpose": "test"}
    shuffle: bool = True
    train_dataset_ratio:int = 6
    query_res_size: str = "100"
    query_match_field: str = "image_class"
    query_match_items: List[str] = [
        "plax", 
        "psaxal", 
        "psaxmv",
        "psaxpml",
        "psaxapical",
        "4chamber",
        "2chamber",
        "3chamber",
        "5chamber",
        "subcostal",
        "ivc",
        ]

@app.get("/")
async def root():
    return {
        "message": "Hello World"
    }

@app.post("/elastic/")
async def getImagePath(query_type: str, option: Option):
    dir_class_list = []
    total_images = 0
    num_dir_min_len = float("inf")
    elastic_object = Elastic(
        os.environ.get("ELASTIC_HOST", "127.0.0.1"),
        os.environ.get("ELASTIC_PORT", "9200"),
        os.environ.get("ELASTIC_INDEX", "test_index")
    )

    for class_name in option.query_match_items:
        
        match query_type:
            
            case "bool":
                query = elastic_object.makeBoolQuery(
                    option.query_res_size, 
                    elastic_object.makeMatchQuery(option.query_match_field, [class_name])
                    )

            case _ :
                return "Invalid query type"
        
        try:
            
            def resPretty(response):
                return [(dir, i["_source"]["image_class"]) for i in response["hits"]["hits"] for dir in i["_source"]["dir"]]

            query_res_tuple = resPretty(elastic_object.elasticGet(query))
            
            total_images += len(query_res_tuple)
            
            num_dir_min_len = min(num_dir_min_len, len(query_res_tuple))

            if query_res_tuple:
                dir_class_list.append(query_res_tuple)
    
        except:
            return str("Error occured")
    
    ai_object = ModelTrain(option)
    
    train_res = ai_object.airun(dir_class_list, num_dir_min_len)

    return [
        {
            "total_num_image": total_images,
            "num_dir_min_len": num_dir_min_len,
            "num_image_class": len(option.query_match_items),
            "image_class": option.query_match_items,
            "epoch_size": option.epoch_size,
            "batch_size": option.batch_size
            },
        train_res
        ]