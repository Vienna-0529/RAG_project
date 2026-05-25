"""
ReAct 模式迭代检索：LLM 根据中间结果自主决定下一轮搜索方向

流程：
  Round 1 — LLM 生成 2-3 个搜索子问题 → BM25 并行检索 → 去重
  Round 2 — LLM 读第一轮结果，判断是否缺信息 → 如有缺口，生成新子问题
  综合   — 合并所有收集到的 chunks，去重后返回
"""

from rag_engine import bm25_engine, all_splits, better_tokenize
from config import AGENTIC as CFG
import ollama


def llm_think(system_prompt: str, user_prompt: str) -> str:
    """调 LLM 做推理决策"""
    resp = ollama.chat(
        model=CFG["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": CFG["strategy_temperature"], "num_predict": 300},
    )
    return resp["message"]["content"].strip()


def search_keywords(keywords: list, top_n: int = 10) -> list:
    """用关键词列表并行检索 BM25，去重合并"""
    all_docs = []
    seen = set()
    for kw in keywords:
        tokens = better_tokenize(kw)
        results = bm25_engine.get_top_n(tokens, all_splits, n=top_n)
        for doc in results:
            if doc.page_content not in seen:
                all_docs.append(doc)
                seen.add(doc.page_content)
    return all_docs


def agentic_retrieve(question: str, max_rounds: int = None, docs_per_query: int = None):
    if max_rounds is None:
        max_rounds = CFG["max_rounds"]
    if docs_per_query is None:
        docs_per_query = CFG["docs_per_query"]
    """
    ReAct 迭代检索：

    参数:
      question: 用户原始问题
      max_rounds: 最大搜索轮次
      docs_per_query: 每个子问题返回的文档数

    返回:
      (collected_docs, reasoning_trace) — 收集的文档列表 + 推理过程
    """
    collected = []
    seen_texts = set()
    trace = []

    # Round 1: 拆解问题 → 并行搜索
    trace.append("[Round 1] 分析问题，生成搜索方向")

    decompose_prompt = f"""你是一个搜索策略专家。用户提出了一个复杂问题，你的任务是把问题拆解
成 2-3 个不同的搜索关键词或子问题，每个从不同角度切入，帮助找到答案。

输出格式：每行一个搜索词，不要编号，不要多余解释。

用户问题：{question}

搜索词："""

    queries_raw = llm_think(
        "你是搜索策略专家。只输出搜索关键词，每行一个，不要任何解释。",
        decompose_prompt,
    )

    queries = [q.strip() for q in queries_raw.split("\n") if q.strip()]
    queries = queries[:3]
    trace.append(f"  生成子查询: {queries}")

    round1_docs = search_keywords(queries, top_n=docs_per_query)
    for doc in round1_docs:
        if doc.page_content not in seen_texts:
            collected.append(doc)
            seen_texts.add(doc.page_content)
    trace.append(f"  收集到 {len(round1_docs)} 个片段（去重后 {len(collected)} 个）")

    # Round 2: 评估缺口 → 补充搜索
    if max_rounds >= 2 and len(collected) > 0:
        trace.append("[Round 2] 评估信息是否充足")

        preview = "\n\n".join([
            f"[{i+1}] {d.page_content[:300]}" for i, d in enumerate(collected[:6])
        ])

        gap_prompt = f"""你是一个搜索策略专家。下面是用户问题和已收集到的片段内容。
请判断：这些片段是否足以回答用户问题？

如果信息已充足，回复"足够"。
如果还需要补充某方面的信息，用一句话描述还需要搜索什么（不要超过15个字）。

用户问题：{question}

已收集片段：
{preview}

判断："""

        gap_answer = llm_think(
            "你是搜索策略专家。判断信息是否充足。只需回复'足够'或一句话描述缺什么。",
            gap_prompt,
        )
        trace.append(f"  LLM 判断: {gap_answer}")

        if "足够" not in gap_answer and len(gap_answer) > 2:
            trace.append(f"  补充搜索: {gap_answer}")
            extra_queries = [gap_answer]
            round2_docs = search_keywords(extra_queries, top_n=docs_per_query)
            new_count = 0
            for doc in round2_docs:
                if doc.page_content not in seen_texts:
                    collected.append(doc)
                    seen_texts.add(doc.page_content)
                    new_count += 1
            trace.append(f"  新增 {new_count} 个片段（总计 {len(collected)} 个）")
    else:
        trace.append("[Round 2] 跳过（已达到最大轮次或未收集到结果）")

    return collected, trace


def agentic_generate(question: str, max_rounds: int = None) -> str:
    """端到端 Agentic RAG：迭代检索 + 综合回答"""
    if max_rounds is None:
        max_rounds = CFG["max_rounds"]
    docs, trace = agentic_retrieve(question, max_rounds=max_rounds)

    if not docs:
        return "抱歉，未找到相关信息。"

    context = "\n\n".join([
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs[:15])
    ])

    system_prompt = (
        "你是一个严谨的文档问答助手。根据提供的【背景资料】回答用户问题。\n"
        "规则：\n"
        "1. 只能基于资料回答，禁止编造。\n"
        "2. 资料不够就诚实说未找到。\n"
        "3. 回答简洁，标注出处编号（如 [1]）。\n"
        "4. 资料中相关信息分散时，请归纳总结形成完整答案。"
    )

    user_prompt = f"【背景资料】\n{context}\n\n【用户问题】{question}\n\n请回答："

    resp = ollama.chat(
        model=CFG["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": CFG["answer_temperature"], "num_ctx": CFG["num_ctx"]},
    )

    answer = resp["message"]["content"].strip()
    return answer
