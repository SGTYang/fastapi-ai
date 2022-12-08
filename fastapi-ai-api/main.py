from ai import ModelTrain
from fastapi import FastAPI
import requests, os, json
from typing import List
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth

'''환경변수 '''
ELASTIC_HOST = os.environ.get("ELASTIC_HOST", "112.220.111.68")
ELASTIC_PORT = os.environ.get("ELASTIC_PORT", "9200")
INDEX = os.environ.get("ELASTIC_INDEX", "test_index")

app = FastAPI()

'''
epoch_size: epoch 크기
batch_size: batch 사이즈
shuffle: 셔플 여부
train_dataset_ratio: train 데이터셋 크기 비율 
query_res_size: 쿼리 요청 사이즈
model_name: efficientnet 모델 종류
query_match_field: match query field값 
query_match_items: 
'''
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
    
''' bool query에 쓰일 match 쿼리 생성 함수'''
def makeMatchQuery(field: str, match_list: list):
    return [{"match": {field: elem}} for elem in match_list]

'''dir, image_class를 포함해서 쿼리하는 bool 쿼리 생성 함수'''
def makeBoolQuery(query_size: int, match_query: list):
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

'''추후 데이터 집게를 위한 aggs 쿼리 생성 함수-미완성'''
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
    
'''
확장: https cert 사용 고려
'''
def elasticGet(query):
    url = f"https://{ELASTIC_HOST}:{ELASTIC_PORT}/{INDEX}/_search?"
    
    header = {'Content-Type': 'application/json'}
    
    return requests.get(
        url,
        headers=header,
        verify=False,
        auth= HTTPBasicAuth("elastic", "xB8nsHxmXqB8J9Dfgzi*"),
        data=json.dumps(query)
    ).json()

@app.post("/elastic/")
async def getImagePath(query_type: str, option: Option):
    dir_class_list = []
    total_images = 0
    num_dir_min_len = float("inf")
    
    ''' 
    가져올 class 개수가 많아지고 쿼리 사이즈가 증가하면 es에서 한번에 쿼리할 수 있는 개수(10,000개)를 초과하기 때문에 class별 나눠서 여러번 쿼리
    -> es 수정하여 쿼리 개수 늘릴수 있지만 es에 걸리는 부하 증가
    '''
    for class_name in option.query_match_items:
        
        '''query 종류에 따라 query body 생성'''
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
            
            ''' 쿼리 결과 배열에 저장 '''
            if query_res_tuple:
                dir_class_list.append(query_res_tuple)
            
        except:
            return str("Error occured")
    
    '''train'''
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