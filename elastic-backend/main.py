
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
    match_field: str = "image_class"
    match: List[str] = ["NA"]
    query: Dict[str, dict] = {
        "image_class": {
            "field": "image_class"
            }
        }

@app.get("/")
async def root():
    return {
        "message": "Hello World"
    }

def makeBoolQuery(match_query):
    return {
        "_source": {
            "includes": [
                "dir",
                "image_class"
            ]
        },
        "query":{
            "bool": {
                "should": match_query
            }
        }
    }

def makeMatchQuery(field: str, match_list: list):
    return [{"match": {field: elem}} for elem in match_list]

def makeAggsQuery(option: Dict[str, str]):
    return {
        "_source": {
            "includes": [
                "dir",
                "image_class"
            ]
        },
        "aggs": {
            "term": option
        },
        "size": 300*11
    }
    
@app.post("/elastic/{query_type}/")
async def getImagePath(query_type: str, option: Option):
    
    match_query = makeMatchQuery(option.match_field, option.match)

    match query_type:
              
        case "bool":
            query = makeBoolQuery(match_query)

        case _ :
            return "Invalid query type"
        
    headers = {'Content-Type': 'application/json'}
    
    response = requests.get(
        f"http://{ELASTIC_HOST}:{ELASTIC_PORT}/{INDEX}/_search",
        headers=headers,
        data=json.dumps(query)
    ).json()
    try:
        class_and_image = {dir:i["_source"]["image_class"] for i in response["hits"]["hits"] for dir in i["_source"]["dir"]}
        image_class = set([i["_source"]["image_class"] for i in response["hits"]["hits"]])
    
        print(image_class, len(class_and_image))
        #await ai.trainMain(option, class_and_image)
    
        return class_and_image

    except Exception as e:
        return str("Error occured")