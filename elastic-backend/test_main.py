import pytest
import main
from httpx import AsyncClient

matchQuery1 = {
        "query":{
            "match":{
                "image_type":{
                    "query": "NA"
                }
            }
        }
    }

matchQuery2 = {
        "query":{
            "match":{
                "image_type":{
                    "query": "NA",
                    "operator": "OR"
                }
            }
        }
    }

def test_makeMatchQuery():
    assert main.makeMatchQuery("image_type", query = "NA") == matchQuery1
    assert main.makeMatchQuery("image_type", query = "NA", operator = "OR") == matchQuery2

@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(app = main.app, base_url="http://127.0.0.1:8000") as ac:
        response = await ac.get("/")
        assert response.status_code >= 200
        assert response.json() == {"message": "Hello World"}