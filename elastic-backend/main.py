from fastapi import FastAPI, Request
import requests, os, json
from typing import Union, Dict, List

ELASTIC_HOST = os.environ.get("ELASTIC_HOST", "192.168.0.212")
ELASTIC_PORT = os.environ.get("ELASTIC_PORT", "9200")
INDEX = os.environ.get("ELASTIC_INDEX", "test_index")

app = FastAPI()

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

@app.post("/AI/{model_name}/")
async def trainModel(model_name: str, images: List[str]):
    print("Model Name:", model_name)
    for image in images:
        print(image)
    return

@app.post("/elastic/{field}/")
async def getImagePath(field: str, option:Dict[str, str] = {"query":"NA"}):
    headers = {'Content-Type': 'application/json'}
    query_res = makeMatchQuery(field, **option)
    response = requests.get(
        f"http://{ELASTIC_HOST}:{ELASTIC_PORT}/{INDEX}/_search?_source_includes=path",
        headers=headers,
        data=json.dumps(query_res)
    ).json()
    print([i["_source"]["path"] for i in response["hits"]["hits"]])
    return response
