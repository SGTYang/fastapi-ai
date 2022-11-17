import ai
from xmlrpc.client import boolean
from fastapi import FastAPI, Request
import requests, os, json
from typing import Union, Dict, List
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth

'''container환경일때 환경변수 사용하기 위해'''
ELASTIC_HOST = os.environ.get("ELASTIC_HOST", "112.220.111.68")
ELASTIC_PORT = os.environ.get("ELASTIC_PORT", "9200")
INDEX = os.environ.get("ELASTIC_INDEX", "test_index")

app = FastAPI()

'''
epoch_size: 에포크 크기
batch_size: 배치 사이즈
shuffle: 셔플 여부
train_dataset_ratio: train 데이터셋 크기 비율 
query_res_size: 쿼리 요청 사이즈
model_name: efficientnet 모델 종류
query_match_field: match query field값 
query_match_items: 
'''
class Option(BaseModel):
    epoch_size: int = 2
    batch_size: int = 128
    shuffle: bool = True
    train_dataset_ratio:int = 6
    query_res_size: str = "100"
    model_name: str = 'efficientnet-b0'
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

'''추후 데이터 집게를 위한 aggs 쿼리 생성 함수 미완성'''
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
http사용해 elasticsearch에 쿼리 요청
https면 header에 보안정보 추가
현재 테스트 중인 es는 보안 옵션을 꺼놓아서 http요청으로만 가능
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
    
@app.post("/elastic/{query_type}/")
async def getImagePath(query_type: str, option: Option):
    dir_class_list = []
    total_images = 0
    num_dir_min_len = float("inf")
    
    ''' 
    가져올 class 개수가 많아지고 쿼리 사이즈가 증가하면 es에서 한번에 쿼리할 수 있는 개수를 초과하기 때문에 class별 나눠서 여러번 쿼리
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
            dir_class_list.append(query_res_tuple)
            
        except Exception as e:
            print(e)
            return str("Error occured")
    
    '''{0: "class_name"} 형식으로 label 생성'''    
    image_label_dict = {val:idx for idx,val in enumerate(option.query_match_items)}
    
    '''쿼리해서 가져온 결과 이미지주소 개수가 달라 개수가 제일 적은 이미지주소가 기준이 되어 그 개수만큼 이미지주소 개수 수정 후 dir와 label mapping'''
    dir_class_dict = {image_dir: image_label_dict[label] for class_elem in dir_class_list for image_dir, label in class_elem[:num_dir_min_len+1]}
    
    train_res = ai.trainMain(option, dir_class_dict)
    
    return [
        {
            "total_num_image": total_images,
            "gained_num_image": len(dir_class_dict),
            "num_dir_min_len": num_dir_min_len,
            "num_image_class": len(option.query_match_items),
            "image_class": option.query_match_items,
            },
        train_res
        ]