# 启动
docker compose up -d
cd backend && pip install -r requirements.txt
cd app && python main.py

# 存储
## Qdrant 向量库存储 + MongoDB 原始文本存储
MongoDB
| id        | lecture_id  | chunk_index | type |
| --------- | ------- | ----------- | ---- |
| chunk_001 | lec_001 | 0           | screen_shot  |
| chunk_002 | lec_001 | 1           | screen_shot  |
| chunk_003 | lec_001 | 2           | screen_shot  |

Qdrant
| id        | lecture_id  | chunk_index | vector    | type |
| --------- | ------- | ----------- | --------- | ---- |
| chunk_001 | lec_001 | 0           | embedding | screen_shot |
| chunk_002 | lec_001 | 1           | embedding | screen_shot |
| chunk_003 | lec_001 | 2           | embedding | screen_shot |

* screen_shot： 将每个截屏内容按 400 token chunk，原始文字保存 MongoDB, 向量库保存 ids
* transcript： 按 lecture 维度  400 token chunk 保存原始文本
* summary： 按 lecture 维度保存原始文本

# 召回
* 双路召回   一路做语义向量 search；另一路直接用queryAPI，按时间窗口捞锚点附近的 chunk；结果合并去重
* Hybrid 多路混合检索（稠密 + 稀疏）
* Rerank 重排在线部署接入检索链路

# 服务化
 将数据库部署服务 提供接口  /insert /delete /search
 docker 一键启动服务

Docker Compose
├── nginx
├── fastapi
├── qdrant
├── mongodb
├── ollama
└── redis（可选）


<!-- screen_shot 识别方案 -->
<!-- 1. 检测黑屏，直接跳过识别
2. phash 检测 -->

<!-- 检索方案 -->
1. rewrite
<!-- {
  "rewritten_query": "如何重置账户密码",
  "hard_constraints": [
    {"field":"tenant_id", "operator":"eq", "value": 88},
    {"field":"doc_type", "operator":"eq", "value": "help_document"}
  ]
} -->

