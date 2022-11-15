# elastic-backend

# elastic search index mapping
dir field mapping 수정 필요 
"dir" 필드 타입을 "text"에서 "nested"로

PUT /test_index
{
  "mappings": {
    "properties": {
      "0008": {
        "properties": {
          "1010": {
            "type": "text"
          },
          "1040": {
            "type": "text"
          },
          "1070": {
            "type": "text"
          },
          "1090": {
            "type": "text"
          },
          "1150": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "1155": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "2142": {
            "type": "integer",
            "ignore_malformed": true
          },
          "2143": {
            "type": "integer",
            "ignore_malformed": true
          },
          "2144": {
            "type": "integer",
            "ignore_malformed": true
          },
          "0008": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "0016": {
            "type": "keyword"
          },
          "0018": {
            "type": "keyword"
          },
          "0020": {
          "type": "date",
          "format": "yyyyMMdd",
          "ignore_malformed": true
          },
          "0021": {
          "type": "date",
          "format": "yyyyMMdd",
          "ignore_malformed": true
          },
          "0023": {
          "type": "date",
          "format": "yyyyMMdd",
          "ignore_malformed": true
          },
          "002A": {
          "type": "date",
          "format": "yyyyMMddHHmmss",
          "ignore_malformed": true
          },
          "0030": {
          "type": "date",
          "format": "HHmmss",
          "ignore_malformed": true
          },
          "0031": {
          "type": "date",
          "format": "HHmmss",
          "ignore_malformed": true
          },
          "0033": {
          "type": "date",
          "format": "HHmmss",
          "ignore_malformed": true
          },
          "0050": {
          "type": "keyword"
          },
          "0060": {
          "type": "keyword"
          },
          "0070": {
          "type": "text"
          },
          "0080": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "0090": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          }
        }
      },
      "0010": {
        "properties": {
          "1000": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "0010": {
            "type": "keyword",
            "fields": {
              "raw": {
                "type": "text"
              }
            }
          },
          "0020": {
            "type": "keyword"
          },
          "0030": {
            "type": "date",
            "format": "yyyyMMdd",
            "ignore_malformed": true
          },
          "0032": {
            "type": "date",
            "format": "HHmmss",
            "ignore_malformed": true
          },
          "0040": {
            "type": "keyword",
            "fields": {
              "raw": {
                "type": "text"
              }
            }
          }
        }
      },
      "0018": {
        "properties": {
          "1020": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "1063": {
            "type": "keyword"
          },
          "1066": {
            "type": "float"
          },
          "1088": {
            "type": "integer",
            "ignore_malformed": true
          },
          "1242": {
            "type": "integer",
            "ignore_malformed": true
          },
          "1244": {
            "type": "short",
            "ignore_malformed": true
          },
          "6012": {
            "type": "short",
            "ignore_malformed": true
          },
          "6014": {
            "type": "short",
            "ignore_malformed": true
          },
          "6016": {
            "type": "long",
            "ignore_malformed": true
          },
          "6018": {
            "type": "long",
            "ignore_malformed": true
          },
          "6020": {
            "type": "long",
            "ignore_malformed": true
          },
          "6022": {
            "type": "long",
            "ignore_malformed": true
          },
          "6024": {
            "type": "short",
            "ignore_malformed": true
          },
          "6026": {
            "type": "short",
            "ignore_malformed": true
          },
          "6028": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "6030": {
            "type": "long",
            "ignore_malformed": true
          },
          "6032": {
            "type": "long"
          },
          "6060": {
            "type": "float"
          },
          "0040": {
            "type": "keyword"
          },
          "0072": {
            "type": "keyword"
          },
          "601A": {
            "type": "long",
            "ignore_malformed": true
          },
          "601C": {
            "type": "long",
            "ignore_malformed": true
          },
          "601E": {
            "type": "long",
            "ignore_malformed": true
          },
          "602A": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          },
          "602C": {
            "type": "float",
            "ignore_malformed": true
          },
          "602E": {
            "type": "float",
            "ignore_malformed": true
          }
        }
      },
      "0020": {
        "properties": {
          "000D": {
            "type": "keyword"
          },
          "000E": {
            "type": "keyword"
          },
          "0010": {
            "type": "keyword"
          },
          "0011": {
            "type": "integer"
          },
          "0013": {
            "type": "integer"
          },
          "0020": {
            "type": "keyword"
          }
        }
      },
      "0032": {
        "properties": {
          "1060": {
            "type": "text"
          }
        }
      },
      "image_class": {
        "type": "keyword"
      },
      "dir": {
        "type": "text"
      }
    }
  }
}