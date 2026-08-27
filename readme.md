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

# 问答路由（按用户问题类型）

`query.txt` 中每条问题统一走：**rewrite → 检索 → 拼 prompt → 回答 → 写 Excel**。  
预过滤始终带上 `course_id` / `quarter` / `lecturer`；数据分 `screen_shot`（课件 OCR）与 `transcript`（字幕）两类。

| 用户问题类型 | rewrite | 检索 | 回答 |
| --- | --- | --- | --- |
| **现在在讲什么**<br>例：`What does this mean now?`、`现在在讲什么` | LLM 输出 `hard_constraints` 含时间戳，如 `{"field":"timestamp","operator":"range","value":"01:22:09"}`；`rewritten_query` 可为占位（检索主要靠时间，不靠关键词） | **仅按时间邻近召回**：播放点 ±2 分钟内 `screen_shot` + `transcript`；不做 dense / BM25 / rerank；`screen_shot` 与 `transcript` 各保留约一半，避免截图挤掉字幕 | `NOW_ANSWER_SYSTEM`：有材料就直接解释当前画面/台词；空结果用固定口语拒答 |
| **普通知识题**<br>例：`What is Fold Left exactly?`、并发相关概念问法 | 扩成技术检索词（同义词、领域术语）；**不加**时间戳约束 | dense（短 query）+ BM25（可含扩充 query）× `screen_shot` / `transcript` 四路召回 → 合并去重 → **rerank**（类型配额保证截图与字幕都保留） | `ANSWER_SYSTEM`：友好小助手语气，只依据检索内容；空结果用固定口语拒答 |
| **带题号 / 定位短语**<br>例：`I don't understand the answer of Question 9` | 保留题号（如 `Question 9`），并扩写检索意图（题干、选项、答案、讲解）；不编造具体题目正文 | 在普通知识题流程上叠加 **短语锚定**：从问题检出 `Label + 数字` → 生成短语 token（如 `Question 9` → `phquestion9`）→ BM25 用 token 锚定真题干页 → 将命中 chunk 前 ~900 字拼入 `q_expand`；`anchor_hits` 强制进入候选后再 rerank | 同上；若材料含 quiz 项，先简述题干/选项再解释 |
| **无命中** | — | 各路上下文均为空，或 rerank 后无保留 | 不交给模型套模板，直接返回口语化拒答（如 *Hmm, I couldn't find…*） |

## 短语 token（Label + 数字）

BM25 词袋会把 `Question` 与 `9` 拆开，易与侧边栏 `[9]`、其它 `Question K` 混淆。当前做法：

1. **入库**：BM25 向量文本 = 原文 + 文中检出的短语 token（`payload.text` 仍为干净原文）
2. **查询**：从用户问题检出同一短语 → 生成相同 token → BM25 锚定 → 可选拼入 `q_expand`

公式：`phrase_token("Question 9")` → `ph` + 去标点空白的小写短语 → `phquestion9`（动态生成，不写死题号）。

维护命令：

```bash
cd backend/app
python main.py --backfill-bm25-phrases   # 旧库只补 BM25 短语 token
python main.py --inspect-phrase "Question 9"  # 对比词袋 BM25 vs 短语 token
python main.py --ingest                  # 全量重建（含短语 token）
python main.py                           # 跑 query.txt 问答
```
