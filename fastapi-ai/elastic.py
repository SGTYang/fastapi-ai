import requests, json
from requests.auth import HTTPBasicAuth

class Elastic:
    def __init__(self, host, port, index):
        self.elastic_host = host
        self.elastic_port = port
        self.elastic_idnex = index
    
    def makeMatchQuery(self, field: str, match_list: list):
        return [{"match": {field: elem}} for elem in match_list]

    def makeBoolQuery(self, query_size: int, match_query: list):
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

    def makeAggsQuery(self):
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

    def elasticGet(self, query):
        url = f"https://{self.elastic_host}:{self.elastic_port}/{self.elastic_idnex}/_search?"
        
        header = {'Content-Type': 'application/json'}
        
        return requests.get(
            url,
            headers=header,
            verify=False,
            auth= HTTPBasicAuth("elastic", "xB8nsHxmXqB8J9Dfgzi*"),
            data=json.dumps(query)
        ).json()