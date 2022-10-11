from fastapi import FastAPI, Request
import requests, os, json
from typing import Union

ELASTIC_HOST = os.environ.get("ELASTIC_HOST", "192.168.0.212")
ELASTIC_PORT = os.environ.get("ELASTIC_PORT", "9200")
INDEX = os.environ.get("ELASTIC_INDEX", "test_index")

app = FastAPI()

@app.get("/")
async def root():
    return {
        "message": "Hello World"
    }

@app.get("/elastic/{item}")
async def elasticGet(item: str, q: Union[str, None] = None):
    query = {
      "query": {
        "match": {
          "path": "202210111037125"
        }
      }
    }
    headers = {'Content-Type': 'application/json'}
    if q:
        return {"item": item, "q": q}
    return requests.get(
        f"http://{ELASTIC_HOST}:{ELASTIC_PORT}/{INDEX}/_search?_source_includes=path",
        headers=headers,
        data=json.dumps(query)
    ).json()