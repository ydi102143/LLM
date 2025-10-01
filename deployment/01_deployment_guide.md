# LLMアプリケーション デプロイメントガイド

このガイドでは、LLMアプリケーションを本番環境にデプロイする方法を説明します。

## 目次

1. [環境準備](#環境準備)
2. [モデルの最適化](#モデルの最適化)
3. [APIサーバーの構築](#apiサーバーの構築)
4. [Docker化](#docker化)
5. [クラウドデプロイメント](#クラウドデプロイメント)
6. [監視とログ](#監視とログ)
7. [セキュリティ](#セキュリティ)
8. [パフォーマンス最適化](#パフォーマンス最適化)

## 環境準備

### 必要なライブラリ

```bash
# 基本ライブラリ
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install fastapi uvicorn
pip install docker
pip install prometheus-client
pip install python-multipart

# 本番環境用
pip install gunicorn
pip install redis
pip install celery
pip install psycopg2-binary
```

### 環境変数の設定

```bash
# .env ファイル
MODEL_NAME=gpt2
MODEL_PATH=./models
API_HOST=0.0.0.0
API_PORT=8000
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:password@localhost/dbname
LOG_LEVEL=INFO
```

## モデルの最適化

### 1. モデルの量子化

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4bit量子化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=quantization_config
)
```

### 2. モデルの最適化

```python
import torch
from torch.jit import script

# TorchScript化
model.eval()
scripted_model = script(model)
scripted_model.save("optimized_model.pt")
```

### 3. モデルのキャッシュ

```python
import pickle
import os

class ModelCache:
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_model(self, model_name):
        cache_path = os.path.join(self.cache_dir, f"{model_name}.pkl")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # モデルを読み込んでキャッシュ
        model = self._load_model(model_name)
        with open(cache_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model
```

## APIサーバーの構築

### 1. FastAPIアプリケーション

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

app = FastAPI(title="LLM API", version="1.0.0")

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# グローバル変数
model = None
tokenizer = None

class TextRequest(BaseModel):
    text: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class TextResponse(BaseModel):
    generated_text: str
    confidence: float
    processing_time: float

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    
    logger.info("モデルを読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("モデルの読み込み完了")

@app.post("/generate", response_model=TextResponse)
async def generate_text(request: TextRequest):
    try:
        import time
        start_time = time.time()
        
        # テキスト生成
        inputs = tokenizer.encode(request.text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        processing_time = time.time() - start_time
        
        return TextResponse(
            generated_text=generated_text,
            confidence=0.8,  # 実際の実装では適切な信頼度を計算
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. 非同期処理

```python
import asyncio
from celery import Celery
import redis

# Celery設定
celery_app = Celery(
    "llm_worker",
    broker="redis://localhost:6379",
    backend="redis://localhost:6379"
)

@celery_app.task
def generate_text_async(text, max_length=100, temperature=0.7):
    # 重い処理を非同期で実行
    # 実際の実装では適切な処理を行う
    return {"generated_text": f"Generated: {text}"}

@app.post("/generate-async")
async def generate_text_async_endpoint(request: TextRequest):
    task = generate_text_async.delay(
        request.text,
        request.max_length,
        request.temperature
    )
    
    return {"task_id": task.id, "status": "processing"}

@app.get("/task/{task_id}")
async def get_task_result(task_id: str):
    task = generate_text_async.AsyncResult(task_id)
    
    if task.ready():
        return {"status": "completed", "result": task.result}
    else:
        return {"status": "processing"}
```

## Docker化

### 1. Dockerfile

```dockerfile
FROM python:3.9-slim

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# Pythonの依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# ポートを公開
EXPOSE 8000

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# アプリケーションを起動
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=gpt2
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - postgres
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=llm_db
      - POSTGRES_USER=llm_user
      - POSTGRES_PASSWORD=llm_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api

volumes:
  redis_data:
  postgres_data:
```

### 3. nginx.conf

```nginx
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## クラウドデプロイメント

### 1. AWS ECS

```yaml
# task-definition.json
{
  "family": "llm-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "llm-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/llm-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_NAME",
          "value": "gpt2"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/llm-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### 2. Google Cloud Run

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/llm-api', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/llm-api']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: ['run', 'deploy', 'llm-api', '--image', 'gcr.io/$PROJECT_ID/llm-api', '--region', 'us-central1']
```

### 3. Azure Container Instances

```yaml
# azure-deploy.yaml
apiVersion: 2018-10-01
location: eastus
name: llm-api
properties:
  containers:
  - name: llm-api
    properties:
      image: your-registry.azurecr.io/llm-api:latest
      ports:
      - port: 8000
      resources:
        requests:
          cpu: 1
          memoryInGb: 2
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
```

## 監視とログ

### 1. Prometheus監視

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# メトリクスの定義
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
ACTIVE_CONNECTIONS = Gauge('api_active_connections', 'Active connections')

@app.middleware("http")
async def monitor_requests(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

# メトリクスサーバーを起動
start_http_server(8001)
```

### 2. ログ設定

```python
import logging
import sys
from pythonjsonlogger import jsonlogger

# ログフォーマッター
logHandler = logging.StreamHandler(sys.stdout)
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)

# ロガー設定
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# カスタムログ
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info("Request started", extra={
        "method": request.method,
        "url": str(request.url),
        "client_ip": request.client.host
    })
    
    response = await call_next(request)
    
    logger.info("Request completed", extra={
        "status_code": response.status_code
    })
    
    return response
```

## セキュリティ

### 1. 認証と認可

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, os.getenv("SECRET_KEY"), algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@app.post("/generate")
async def generate_text(request: TextRequest, user=Depends(verify_token)):
    # 認証されたユーザーのみアクセス可能
    pass
```

### 2. レート制限

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/generate")
@limiter.limit("10/minute")
async def generate_text(request: Request, text_request: TextRequest):
    # 1分間に10回まで制限
    pass
```

### 3. 入力検証

```python
from pydantic import BaseModel, validator
import re

class TextRequest(BaseModel):
    text: str
    max_length: int = 100
    temperature: float = 0.7
    
    @validator('text')
    def validate_text(cls, v):
        if len(v) > 1000:
            raise ValueError('Text too long')
        if not re.match(r'^[a-zA-Z0-9\s.,!?]+$', v):
            raise ValueError('Invalid characters')
        return v
    
    @validator('max_length')
    def validate_max_length(cls, v):
        if v > 500:
            raise ValueError('Max length too large')
        return v
```

## パフォーマンス最適化

### 1. キャッシュ戦略

```python
import redis
import json
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cache_key(text, max_length, temperature):
    key_data = f"{text}_{max_length}_{temperature}"
    return hashlib.md5(key_data.encode()).hexdigest()

@app.post("/generate")
async def generate_text(request: TextRequest):
    cache_key = get_cache_key(request.text, request.max_length, request.temperature)
    
    # キャッシュから取得
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result)
    
    # 生成実行
    result = generate_text_model(request)
    
    # キャッシュに保存（1時間）
    redis_client.setex(cache_key, 3600, json.dumps(result))
    
    return result
```

### 2. 非同期処理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def generate_text_async(request: TextRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        generate_text_model, 
        request
    )
    return result
```

### 3. 負荷分散

```python
from fastapi import FastAPI
import uvicorn
import multiprocessing

def create_app():
    app = FastAPI()
    # アプリケーション設定
    return app

if __name__ == "__main__":
    # マルチプロセスで起動
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=multiprocessing.cpu_count()
    )
```

## まとめ

このガイドに従って、LLMアプリケーションを本番環境にデプロイできます。各段階で適切な設定と最適化を行うことで、安定した高パフォーマンスなサービスを構築できます。

### 次のステップ

1. 監視ダッシュボードの構築
2. 自動スケーリングの設定
3. 災害復旧計画の策定
4. セキュリティ監査の実施
5. パフォーマンステストの実行
