"""
全局配置文件 — 所有可调参数集中管理

使用: from config import INGEST, RAG, QUERY_REWRITE, AGENTIC, EVALUATE
"""

#全局语言开关 ("zh" | "en")

LANG = "zh"

#  1. 数据摄入 (ingest.py)
INGEST = {
    # ── 切片 ──
    "chunk_size": 500,          # 切片最大字符数
    "overlap_chars": 100,       # 相邻切片重叠字符数
    "separators": {
        "zh": ["\n\n", "\n", "。", "！", "？", "；", "，", " "],
        "en": ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
    },

    # ── Embedding 模型 ──
    "embedding_model": {
        "zh": "BAAI/bge-large-zh-v1.5",
        "en": "BAAI/bge-large-en-v1.5",
    },

    # ── 向量库 ──
    "vector_db_dir": "./chroma_db",

    # ── 语料存储 ──
    "corpus_file": "bm25_corpus.pkl",

    # —— 文本来源 ——
    # 支持 PDF / TXT / MD / DOCX / 网页 URL，根据后缀自动选 Loader
    "sources": ["./test.pdf"],
}

#  2. 检索 & 生成 (rag_engine.py)
RAG = {
    # ── 生成模型 ──
    "model": "qwen2.5:7b",
    # 备选: "qwen2:7b"

    # ── 生成参数 ──
    "temperature": 0.2,         # 越低越确定，事实问答用 0.1~0.3
    "top_p": 0.8,
    "num_ctx": 8192,            # 上下文窗口 (token 数)

    # ── 检索数量 ──
    "retrieval_k": 15,          # 默认返回 Top-K

    # ── BM25 ──
    "bm25_k1": 1.2,             # TF 饱和度 (1.2~1.5)，越低稀有词越重要
    "bm25_b": 0.75,          # 长度归一化 (0.75 是经典值)，越高越偏好短文档

    # ── 混合检索融合权重 ──
    "vector_weight": 0.3,       # 向量权重 (0~1)
    "bm25_weight": 0.7,         # BM25 权重 (0~1)

    # ── Reranker ──
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "reranker_pool_size": 20,   # 粗召回候选数（减半，延迟从~90s降到~45s）

    # ── 上下文扩展 ──
    "expand_before": 1,         # 前邻居数
    "expand_after": 1,          # 后邻居数
    "expand_min_chars": 200,    # 短于此长度的 chunk 才触发扩展
}

#  3. 查询改写 (query_rewriter.py)
QUERY_REWRITE = {
    "model": "qwen2.5:7b",      # 指令遵循比 qwen2:7b 强，改写不会跑偏成回答
    "temperature": 0.1,          # 改写要确定性，不能有随机性
    "num_predict": 100,          # 改写后问题通常 30 字以内
}


#  4. Agentic 检索 (agentic_search.py)
AGENTIC = {
    "model": "qwen2.5:7b",
    "max_rounds": 2,            # 最大搜索轮次
    "docs_per_query": 8,        # 每个子查询返回数
    "strategy_temperature": 0.1, # 搜索策略温度（需低，保证稳定）
    "answer_temperature": 0.2,   # 回答生成温度
    "num_ctx": 8192,
}

#  5. 评估 (evaluate.py)
EVALUATE = {
    "test_set_path": "./data/test_set.json",
    "multi_turn_path": "./data/multi_turn_test.json",
    "k_values": [3, 5, 10],     # 评估的 K 值
    "relevance_mode": "loose",  # "strict"=全命中, "loose"=半数命中
}
