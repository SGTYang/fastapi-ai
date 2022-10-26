from xmlrpc.client import boolean
from fastapi import FastAPI, Request
import requests, os, json
from typing import Union, Dict, List
from pydantic import BaseModel

ELASTIC_HOST = os.environ.get("ELASTIC_HOST", "112.220.111.68")
ELASTIC_PORT = os.environ.get("ELASTIC_PORT", "9200")
INDEX = os.environ.get("ELASTIC_INDEX", "test_index")

app = FastAPI()

class Option(BaseModel):
    epoch_size: int = 1
    batch_size: int = 1
    shuffle: bool = False
    num_workers: int = 0
    collate_fn: int = None
    pin_memory: bool = None
    drop_last: bool = False
    timeout: int = 0
    worker_init_fn: int = None
    prefetch_factor: int = 2
    persistent_workers: bool = False
    model_name: str = 'efficientnet-b7'
    query:Dict[str, dict] = {
        "terms": {"field": "image_type.keyword"}
        }

@app.get("/")
async def root():
    return {
        "message": "Hello World"
    }

def makeMatchQuery(field: str, **option: Dict[str, str]):
    return {
        "_source": {
            "includes": [
                "path",
                "image_type"
            ]
        },
        "query":{
            "match":{
                field: option
            }
        },
        "size": 300
    }

def makeAggsQuery(field: str, **option: Dict[str, str]):
    return {
        "_source": {
            "includes": [
                "path",
                "image_type"
            ]
        },
        "aggs": {
            field: option
        },
        "size": 300*11
    }
    

@app.post("/elastic/{field}/")
async def getImagePath(field: str, option: Option):
    
    headers = {'Content-Type': 'application/json'}
    
    query_res = makeMatchQuery(field, **option.query)
    aggs_res = makeAggsQuery(field, **option.query)
    
    response = requests.get(
        f"http://{ELASTIC_HOST}:{ELASTIC_PORT}/{INDEX}/_search",
        headers=headers,
        data=json.dumps(aggs_res)
    ).json()
    
    class_and_image = {i["_source"]["path"]: i["_source"]["image_type"] for i in response["hits"]["hits"]}
    image_class = set([i["_source"]["image_type"] for i in response["hits"]["hits"]])
    image_class_list = [i["_source"]["image_type"] for i in response["hits"]["hits"]]
    image_list = [i["_source"]["path"] for i in response["hits"]["hits"]]
    
    print(image_class, len(class_and_image))
    #ai.trainMain(option, image_list)
    
    return response
