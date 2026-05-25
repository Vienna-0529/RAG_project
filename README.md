# RAG 智能问答系统

基于 BM25 + BGE 向量检索 + Cross-Encoder Reranker 的本地 RAG 管道，支持多格式文档摄入、三路自动路由和 Agentic ReAct 多跳检索。

## 架构

```
用户问题
  → 查询改写（多轮指代消解）
  → 三路路由（Agentic / Reranker / BM25）
  → BM25 + BGE 向量双路粗召回 → RRF 融合
  → Cross-Encoder 精排（可选）
  → Ollama LLM 生成回答
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 拉取模型
ollama pull qwen2.5:7b
ollama pull llava:7b      # 可选，图片提取用

# 3. 准备 PDF，修改 config.py 中 sources
"sources": ["./your_file.pdf"],

# 4. 摄入数据
python ingest.py

# 5. 启动问答
streamlit run app.py

# 6. 评估检索质量
python evaluate.py
```

## 评估结果

| 配置 | MRR | Hit |
|------|-----|-----|
| 纯 BM25 | 0.44 | 75% |
| BM25 + Reranker | 0.60 | 75% |

## 项目结构

| 文件 | 作用 |
|------|------|
| `config.py` | 全局参数（语言、切片、检索、模型）|
| `ingest.py` | PDF/TXT/MD/DOCX/网页摄入 + 向量化 |
| `rag_engine.py` | 混合检索、Reranker、Agentic 路由、LLM 生成 |
| `query_rewriter.py` | 多轮对话指代消解查询改写 |
| `agentic_search.py` | ReAct 模式多跳检索 |
| `evaluate.py` | 检索质量评估（Recall@K / MRR / Hit Rate）|
| `app.py` | Streamlit 聊天界面 |

## 技术栈

LangChain / ChromaDB / BM25 / BGE-large-zh-v1.5 / bge-reranker-v2-m3 / Ollama (qwen2.5:7b) / Streamlit
