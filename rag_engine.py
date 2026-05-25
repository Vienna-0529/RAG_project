import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import jieba
from jieba import analyse
import pickle
import ollama
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from query_rewriter import smart_rewrite
from collections import defaultdict
from config import RAG as CFG, INGEST, LANG

# Reranker（延迟加载，首次使用时初始化）
_reranker_model = None


def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(CFG["reranker_model"])
    return _reranker_model


def rerank(question: str, docs: list, top_k: int = 10) -> list:
    """
    Cross-Encoder 重排序：同时读 (问题, chunk) 对，输出相关性分数。
    比 Bi-Encoder 多一层深层语义匹配。
    """
    if len(docs) <= top_k:
        return docs
    model = _get_reranker()
    pairs = [(question, doc.page_content) for doc in docs]
    scores = model.predict(pairs, show_progress_bar=False)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]

_STOPWORDS_ZH = {
    '的', '了', '和', '是', '在', '有', '一', '上', '人', '这',
    '中', '大', '为', '以', '个', '地', '我', '们', '来', '到',
    '对', '于', '从', '他', '她', '它', '可', '也', '都', '很',
}
_STOPWORDS_EN = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
    'and', 'or', 'but', 'not', 'this', 'that', 'it', 'its', 'we',
    'you', 'he', 'she', 'they', 'have', 'has', 'had', 'do', 'does',
}


def better_tokenize(text):
    if LANG == "en":
        words = text.lower().split()
        return [w for w in words if len(w) > 1 and w not in _STOPWORDS_EN]
    words = jieba.cut(text, cut_all=False)
    return [w for w in words if len(w) > 1 and w not in _STOPWORDS_ZH]

embeddings = HuggingFaceEmbeddings(
    model_name=INGEST["embedding_model"][LANG],
    model_kwargs={"local_files_only": True},
)
vectorstore = Chroma(persist_directory=INGEST["vector_db_dir"], embedding_function=embeddings)

with open(INGEST["corpus_file"], "rb") as f:
    all_splits = pickle.load(f)

    doc_chunks_map = defaultdict(list)
    for doc in all_splits:
        doc_id = doc.metadata.get('doc_id')
        if doc_id:
            doc_chunks_map[doc_id].append(doc)

    for doc_id in doc_chunks_map:
        doc_chunks_map[doc_id].sort(key=lambda d: d.metadata.get('chunk_index', 0))
        
tokenized_corpus = [better_tokenize(doc.page_content) for doc in all_splits]
bm25_engine = BM25Okapi(tokenized_corpus, k1=CFG["bm25_k1"])

def generate_response_with_rewrite(question, history=None):
    rewritten = smart_rewrite(question, history)
    return generate_response(rewritten)

def get_hybrid_context_with_docs(question, k=None, vector_weight=None, bm25_weight=None,
                                  use_rerank=False):
    if k is None:
        k = CFG["retrieval_k"]
    if vector_weight is None:
        vector_weight = CFG["vector_weight"]
    if bm25_weight is None:
        bm25_weight = CFG["bm25_weight"]

    # 粗召回
    fetch_k = max(k, CFG["reranker_pool_size"]) if use_rerank else k

    vector_results = vectorstore.similarity_search(question, k=fetch_k)
    tokenized_query = better_tokenize(question)
    bm25_results = bm25_engine.get_top_n(tokenized_query, all_splits, n=fetch_k)

    scored_docs = []
    for i, doc in enumerate(vector_results):
        score = vector_weight * (1 - i / fetch_k) if fetch_k > 0 else 0
        scored_docs.append((doc, score, "vector"))
    for i, doc in enumerate(bm25_results):
        score = bm25_weight * (1 - i / fetch_k) if fetch_k > 0 else 0
        scored_docs.append((doc, score, "bm25"))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    combined_docs = []
    seen = set()
    pool_size = fetch_k * 2 if use_rerank else k
    for doc, score, source in scored_docs:
        if doc.page_content not in seen:
            combined_docs.append(doc)
            seen.add(doc.page_content)
            if len(combined_docs) >= pool_size:
                break

    if use_rerank:
        combined_docs = rerank(question, combined_docs, top_k=k)

    context = "\n\n".join([doc.page_content for doc in combined_docs[:k]])
    return context, combined_docs[:k]

def expand_context(retrieved_docs, before=1, after=1, min_chars=None):
    """扩展短 chunk 的上下文邻居，保持检索排名，去重"""
    if not retrieved_docs:
        return []

    if min_chars is None:
        min_chars = CFG.get("expand_min_chars", 200)

    seen = set()
    expanded = []

    for doc in retrieved_docs:
        doc_id = doc.metadata.get('doc_id')
        idx = doc.metadata.get('chunk_index')

        # 无元数据 或 chunk 足够长 → 原样输出，不扩展
        if doc_id is None or idx is None or len(doc.page_content) >= min_chars:
            if doc.page_content not in seen:
                expanded.append(doc.page_content)
                seen.add(doc.page_content)
            continue

        # 短 chunk：扩展前后邻居
        chunk_list = doc_chunks_map.get(doc_id, [])
        offsets = [0] + [o for o in range(-before, after + 1) if o != 0]
        for offset in offsets:
            neighbor_idx = idx + offset
            if neighbor_idx < 0:
                continue
            for d in chunk_list:
                if d.metadata.get('chunk_index') == neighbor_idx:
                    if d.page_content not in seen:
                        expanded.append(d.page_content)
                        seen.add(d.page_content)
                    break

    return expanded

def _route_question(question: str) -> str:
    """
    三路自动路由：
      "agentic" — 多跳因果推理，需拆解子问题搜索
      "reranker" — 中等复杂度，CrossEncoder 精排
      "bm25" — 简单事实，关键词直出
    """
    causal_core = any(kw in question for kw in [
        "为什么", "原因", "怎么导致", "如何导致", "起因", "怎么才能",
    ])
    multi_hop = causal_core and (
        len(question) > 12 or "和" in question or "与" in question
    )
    is_pure_fact = any(p in question for p in [
        "叫什么", "是谁", "名字", "在哪里", "哪一年", "多少", "几点", "什么时候",
    ])

    if multi_hop and not is_pure_fact:
        return "agentic"
    elif is_pure_fact:
        return "bm25"
    else:
        return "reranker"


def generate_response(
        question, use_expand=True, expand_before=None, expand_after=None,
                      model=None, k=None, use_agentic="auto", use_rerank="auto"):
    """
    参数:
      use_agentic: True=强制Agentic, False=强制标准, "auto"=自动判断
      use_rerank: True=强制Reranker, False=不启用, "auto"=自动判断
    """
    if model is None:
        model = CFG["model"]
    if k is None:
        k = CFG["retrieval_k"]
    if expand_before is None:
        expand_before = CFG["expand_before"]
    if expand_after is None:
        expand_after = CFG["expand_after"]
    route = _route_question(question)
    if use_agentic == "auto":
        use_agentic = (route == "agentic")
    if use_rerank == "auto":
        use_rerank = (route == "reranker")

    if use_agentic:
        from agentic_search import agentic_retrieve
        docs, trace = agentic_retrieve(question)
        if not docs:
            return "抱歉，未找到相关信息。", []
        context = "\n\n".join([
            f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:15])
        ])
        source_docs = docs[:15]
    else:
        context, source_docs = get_hybrid_context_with_docs(
            question, k=k, use_rerank=use_rerank
        )
        if use_expand:
            expanded_texts = expand_context(source_docs, before=expand_before, after=expand_after)
            context = "\n\n".join(expanded_texts)

    system_prompt = (
        "你是一个严谨的文档问答助手。你的任务是根据下方【背景资料】回答用户问题。\n"
        "规则：\n"
        "1. 只能基于【背景资料】中的内容回答，禁止使用外部知识或编造。\n"
        "2. 如果资料中没有相关信息，直接回复\"抱歉，在现有文档中未找到相关记载。\"\n"
        "3. 回答简洁直接，不重复问题，不说\"根据资料显示\"等开场白。\n"
        "4. 如果资料中有多个相关点，分条列出，并标注出处片段编号（如 [1], [2]）。\n"
        "5. 如果答案需要综合多个片段的信息，请归纳总结，但要确保每个结论都能在资料中找到依据。"
    )

    user_prompt = (
        f"【背景资料】\n{context}\n\n"
        f"【用户问题】{question}\n\n"
        "请回答："
    )

    # 调用 Ollama
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                options={
                    "temperature": CFG["temperature"],
                    "top_p": CFG["top_p"],
                    "num_ctx": CFG["num_ctx"],
                }
            )
            return response['message']['content'], source_docs
        except RuntimeError as e:
            if "client has been closed" in str(e):
                print(f"Ollama客户端关闭，重试第{attempt+1}次...")
                if attempt == max_retries - 1:
                    raise e
                continue
            else:
                raise e
