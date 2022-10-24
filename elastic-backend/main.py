import ai

from xmlrpc.client import boolean
from fastapi import FastAPI, Request
import requests, os, json
from typing import Union, Dict, List
from pydantic import BaseModel

ELASTIC_HOST = os.environ.get("ELASTIC_HOST", "192.168.0.212")
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
    query:Dict[str, str] = {
        "_source": {
            "includes": [
                "path",
                "image_type"
            ]
        },
        "query":"NA"
        }

@app.get("/")
async def root():
    return {
        "message": "Hello World"
    }

def makeMatchQuery(field: str, **option: Dict[str, str]):
    return {
        "query":{
            "match":{
                field: option
            }
        }
    }

@app.post("/elastic/{field}/")
async def getImagePath(field: str, option: Option):
    
    headers = {'Content-Type': 'application/json'}
    
    query_res = makeMatchQuery(field, **option.query)
    
    response = requests.get(
        f"http://{ELASTIC_HOST}:{ELASTIC_PORT}/{INDEX}/_search",
        headers=headers,
        data=json.dumps(query_res)
    ).json()
    
    classes = set([i["_source"]["image_type"] for i in response["hits"]["hits"]])
    image_list = [i["_source"]["path"] for i in response["hits"]["hits"]]
    
    print(classes)
    #ai.trainMain(option, image_list)
    
    return response