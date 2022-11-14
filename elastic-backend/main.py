import ai
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
    epoch_size: int = 3
    batch_size: int = 16
    shuffle: bool = False
    drop_last: bool = False
    train_dataset_ratio:int = 6
    query_res_size: str = "100"
    model_name: str = 'efficientnet-b0'
    query_match_field: str = "image_class"
    query_match_items: List[str] = [
        "plax", 
        "psaxal", 
        "psaxmv",
        "psaxpml",
        "psaxapical"
        ]

@app.get("/")
async def root():
    return {
        "message": "Hello World"
    }

''' bool query에 쓰일 match 쿼리 생성 함수'''
def makeMatchQuery(field: str, match_list: list):
    return [{"match": {field: elem}} for elem in match_list]

'''dir, image_class를 포함해서 쿼리하는 bool 쿼리 생성 함수'''
def makeBoolQuery(query_size, match_query):
    return {
        "_source": {
            "includes": [
                "dir",
                "image_class"
            ]
        },
        "query":{
            "bool": {
                "must": match_query
            }
        },
        "size": query_size
    }

'''추후 데이터 집게를 위한 aggs 쿼리 생성 함수'''
def makeAggsQuery():
    return{
        "_source": {
            "includes": [
                "dir",
                "image_class"
            ]
        },
        "aggs":{
            
        }
    }
    
''' http사용해 elasticsearch에 쿼리 날리기'''
def elasticGet(query):
    url = f"http://{ELASTIC_HOST}:{ELASTIC_PORT}/{INDEX}/_search?"
    
    header = {'Content-Type': 'application/json'}
    
    return requests.get(
        url,
        headers=header,
        data=json.dumps(query)
    ).json()
    
@app.post("/elastic/{query_type}/")
async def getImagePath(query_type: str, option: Option):
    
    dir_class_list = []
    total_images = 0
    num_dir_min_len = float("inf")
    
    for class_name in option.query_match_items:

        match query_type:
                
            case "bool":
                query = makeBoolQuery(option.query_res_size, makeMatchQuery(option.query_match_field, [class_name]))

            case _ :
                return "Invalid query type"
    
        try:
            def resPretty(response):
                return [(dir, i["_source"]["image_class"]) for i in response["hits"]["hits"] for dir in i["_source"]["dir"]]
            
            query_res_tuple = resPretty(elasticGet(query))
            total_images += len(query_res_tuple)
            num_dir_min_len = min(num_dir_min_len, len(query_res_tuple))
            
            dir_class_list.append(query_res_tuple)
            
        except Exception as e:
            print(e)
            return str("Error occured")
        
    image_label_dict = {val:idx for idx,val in enumerate(option.query_match_items)}

    dir_class_dict = {image_dir: image_label_dict[label] for class_elem in dir_class_list for image_dir, label in class_elem[:num_dir_min_len+1]}
    
    train_res = ai.trainMain(option, dir_class_dict)
    
    return [
        {
            "total_num_image": total_images,
            "gained_num_image": len(dir_class_dict),
            }, 
        {
            "num_dir_min_len": num_dir_min_len,
            "num_image_class": len(option.query_match_items),
            "image_class": option.query_match_items,
            },
        train_res
        ]