# Fastapi-ai-API 
  1. http://host:10300/elastic/으로 post요청이 들어오면 elasticsearch에서 요청에 맞게 데이터 불러옴
  2. 불러온 데이터를 변환후 모델 학습
  3. 모델 학습에 쓰인 파라미터 및 학습 중간에 나오는 메트릭은 mlflow에 저장
  4. 학습이 끝나면 모델은 mlflow artifact 서버 위치에 저장 
  5. post요청이 들어온 곳으로 학습 결과 return

  # Mlflow에 저장하고 있는 metrics
    "f1score"
    "accuracy"
    "loss"

# Mlflow에 저장하고 있는 parameters
    "batch_size"
    "dataset_size"
    "epoch_size"
    "query_match_items"
    "train_dataset_ratio"