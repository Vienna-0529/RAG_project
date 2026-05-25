import json
import sys
from config import EVALUATE as CFG, RAG
from query_rewriter import smart_rewrite
from rag_engine import (
    vectorstore, bm25_engine, all_splits, better_tokenize,
    get_hybrid_context_with_docs, expand_context,
)


def load_test_set(path: str = CFG["test_set_path"]) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_relevant(chunk_text: str, keywords: list[str], mode: str = CFG["relevance_mode"]) -> bool:
    if not keywords:
        return False
    matched = sum(1 for kw in keywords if kw in chunk_text)
    return matched == len(keywords) if mode == "strict" else matched >= max(1, len(keywords) // 2 + 1)


def compute_metrics(results: list[tuple], k_values: list[int] = CFG["k_values"]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    metrics = {f"recall@{k}": 0.0 for k in k_values}
    rr_sum = 0.0
    hits = 0

    for _question, chunks, keywords in results:
        first_rank = None
        for rank, chunk in enumerate(chunks, start=1):
            if is_relevant(chunk, keywords):
                first_rank = rank
                break
        rr_sum += 1.0 / first_rank if first_rank else 0
        hits += 1 if first_rank else 0

        total_relevant = sum(1 for s in all_splits if is_relevant(s.page_content, keywords))
        for k in k_values:
            relevant_in_k = sum(1 for c in chunks[:k] if is_relevant(c, keywords))
            metrics[f"recall@{k}"] += relevant_in_k / max(total_relevant, 1)

    metrics["mrr"] = rr_sum / n
    metrics["hit_rate"] = hits / n
    for k in k_values:
        metrics[f"recall@{k}"] /= n
    return metrics


def evaluate_config(name: str, retrieval_fn, test_set: list[dict], k: int = 10) -> dict:
    results = []
    for item in test_set:
        chunks = retrieval_fn(item["question"], k=k)
        results.append((item["question"], chunks, item["relevant_keywords"]))
    m = compute_metrics(results)
    m["config"] = name
    return m


# ── 检索配置 ──

def retrieval_vector(q: str, k: int = 10) -> list[str]:
    return [d.page_content for d in vectorstore.similarity_search(q, k=k)]

def retrieval_bm25(q: str, k: int = 10) -> list[str]:
    tokens = better_tokenize(q)
    return [d.page_content for d in bm25_engine.get_top_n(tokens, all_splits, n=k)]

def make_hybrid(vw: float, bw: float):
    def _run(q: str, k: int = 10) -> list[str]:
        _, docs = get_hybrid_context_with_docs(q, k=k, vector_weight=vw, bm25_weight=bw)
        return [d.page_content for d in docs]
    return _run

def make_hybrid_expand(vw: float, bw: float):
    def _run(q: str, k: int = 10) -> list[str]:
        _, docs = get_hybrid_context_with_docs(q, k=k, vector_weight=vw, bm25_weight=bw)
        return expand_context(docs, before=RAG["expand_before"], after=RAG["expand_after"],
                             min_chars=RAG["expand_min_chars"])
    return _run

def make_rerank(vw: float, bw: float):
    def _run(q: str, k: int = 10) -> list[str]:
        _, docs = get_hybrid_context_with_docs(q, k=k, vector_weight=vw, bm25_weight=bw,
                                                use_rerank=True)
        return [d.page_content for d in docs]
    return _run

def with_bm25_rewrite(q: str, k: int = 10) -> list[str]:
    rewritten = smart_rewrite(q, history=None)
    return retrieval_bm25(rewritten, k=k)


# ── 打印 ──

def print_table(metrics_list: list[dict], k_values: list[int] = CFG["k_values"]):
    header = f"{'配置':<22} " + " ".join(f"{'Recall@'+str(k):>9}" for k in k_values) + f" {'MRR':>8} {'Hit':>8}"
    print("\n" + header)
    print("-" * len(header))

    best = {}
    for k in k_values:
        best[f"recall@{k}"] = max(m[f"recall@{k}"] for m in metrics_list)
    best["mrr"] = max(m["mrr"] for m in metrics_list)
    best["hit_rate"] = max(m["hit_rate"] for m in metrics_list)

    for m in metrics_list:
        def mark(key):
            return f"{m[key]:>.4f} *" if abs(m[key] - best[key]) < 1e-9 else f"{m[key]:>.4f}  "
        line = f"{m['config']:<22} " + " ".join(mark(f"recall@{k}") for k in k_values)
        line += f" {mark('mrr'):>9} {mark('hit_rate'):>9}"
        print(line)
    print("  * 最优")


# ── 主入口 ──

def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print("=" * 60)
    print("  RAG 检索评估")
    print("=" * 60)

    test_set = load_test_set()
    print(f"测试集: {len(test_set)} 题")

    if verbose:
        print("\n逐题详情:")
        for item in test_set:
            chunks = make_hybrid(RAG["vector_weight"], RAG["bm25_weight"])(item["question"])
            first = next((i+1 for i, c in enumerate(chunks) if is_relevant(c, item["relevant_keywords"])), None)
            print(f"  [{'命中 #'+str(first) if first else '未命中'}] {item['question']}")

    # 单轮评估 — 工厂函数批量生成权重变体
    configs = [
        ("纯向量",              retrieval_vector),
        ("纯BM25",              retrieval_bm25),
        ("BM25+查询改写",        with_bm25_rewrite),
    ]
    # 混合检索权重扫描：工厂函数一行生成 5 组变体
    for vw in [0.0, 0.3, 0.5, 0.7, 1.0]:
        bw = round(1.0 - vw, 1)
        configs.append((f"混合 V{vw}+B{bw}", make_hybrid(vw, bw)))
    # 混合 + 扩展：用默认权重跑一次
    configs.append((f"混合+扩展 V0.3+B0.7", make_hybrid_expand(0.3, 0.7)))
    # Reranker：BM25 粗召回 + Cross-Encoder 精排
    configs.append(("BM25+Reranker", make_rerank(0.0, 1.0)))

    metrics_list = []
    for name, fn in configs:
        m = evaluate_config(name, fn, test_set)
        metrics_list.append(m)
        print(f"  {name:<22} MRR={m['mrr']:.4f}  Hit={m['hit_rate']:.2%}")

    print_table(metrics_list)

    # 失败案例
    fail = 0
    for item in test_set:
        chunks = make_hybrid(0.3, 0.7)(item["question"])
        if not any(is_relevant(c, item["relevant_keywords"]) for c in chunks[:5]):
            if fail < 5:
                print(f"  x \"{item['question']}\"")
            fail += 1
    if fail == 0:
        print("  全部命中")

    # 多轮评估
    print("\n" + "=" * 60)
    print("  多轮 — 查询改写")
    print("=" * 60)

    try:
        multi = load_test_set(CFG["multi_turn_path"])
        print(f"测试集: {len(multi)} 个场景")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  不可用: {e}")
        return

    results_no, results_rw = [], []
    for item in multi:
        q, h, kw = item["question"], item["history"], item["relevant_keywords"]

        chunks_no = retrieval_bm25(q)
        results_no.append((q, chunks_no, kw))

        rewritten = smart_rewrite(q, history=h)
        chunks_rw = retrieval_bm25(rewritten)
        results_rw.append((q, chunks_rw, kw))

    m_no = compute_metrics(results_no)
    m_rw = compute_metrics(results_rw)
    kv = CFG["k_values"]
    header = f"  {'':<14} " + " ".join(f"{'Recall@'+str(k):>9}" for k in kv) + f" {'MRR':>8} {'Hit':>8}"
    print(header)
    line_no = f"  {'无改写':<14} " + " ".join(f"{m_no[f'recall@{k}']:>9.4f}" for k in kv)
    line_no += f" {m_no['mrr']:>8.4f} {m_no['hit_rate']:>9.2%}"
    print(line_no)
    line_rw = f"  {'有改写':<14} " + " ".join(f"{m_rw[f'recall@{k}']:>9.4f}" for k in kv)
    line_rw += f" {m_rw['mrr']:>8.4f} {m_rw['hit_rate']:>9.2%}"
    print(line_rw)
    change = (m_rw["mrr"] - m_no["mrr"]) / max(m_no["mrr"], 0.001) * 100
    print(f"  → MRR {'↑' if change > 0 else '↓'} {abs(change):.1f}%")


if __name__ == "__main__":
    main()
