import ollama
from typing import List, Dict, Optional
from config import QUERY_REWRITE as CFG,LANG

_SYSTEM_PROMPTS = {
    "zh": (
        "把问题中的代词替换为对话历史中对应的人名或事物，改写为独立完整的问题。"
        "就近原则：优先指代助手回复中讨论的人。只输出改写后的问题，以？结尾。"
        "\n示例：历史→用户问\"福贵的儿子叫什么？\"助手答\"有庆。\""
        "用户问\"他后来怎么了？\"→输出：有庆后来怎么了？"
    ),
    "en": (
        "Replace pronouns with the corresponding names or things from chat history. "
        "Pronouns refer to whom the assistant discussed last. Output only the rewritten question, ending with ?."
        "\nExample: History→User:\"Who is Bob's father?\" Assistant:\"Bob's father is John.\""
        " User:\"What is his attitude?\"→Output: What is John's attitude?"
    ),
}

SYSTEM_PROMPT = _SYSTEM_PROMPTS.get(LANG, _SYSTEM_PROMPTS["zh"])

def rewrite_query(
    original_query: str,
    history: Optional[List[Dict[str, str]]] = None,
    model: str = None,
    temperature: float = None,
) -> str:
    """将模糊问题改写为完整的检索查询"""
    if model is None:
        model = CFG["model"]
    if temperature is None:
        temperature = CFG["temperature"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": original_query})

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature, "num_predict": CFG["num_predict"]}
            )
            rewritten = response['message']['content'].strip()
            # 安全兜底：改写结果必须是问句（以？结尾）且不能太长（防止模型在回答问题）
            if rewritten.endswith("？") and len(rewritten) <= 80:
                return rewritten
            return original_query
        except RuntimeError as e:
            if "client has been closed" in str(e):
                print(f"Ollama客户端关闭，重试第{attempt+1}次...")
                if attempt == max_retries - 1:
                    raise e
                continue
            else:
                raise e

# ── 判断是否需要改写 ─────────────────────────────────────────

def needs_rewrite(question: str, history: Optional[List[Dict[str, str]]] = None) -> bool:
    """有历史时，检测问题是否含指代词、模糊表述或省略追问，需要改写补全"""
    if not history:
        return False

    ambiguous_patterns = [
        "他", "她", "它", "他们", "她们", "它们",
        "这个", "那个", "这些", "那些", "这件事", "那件事",
        "这样", "那样", "怎么样", "什么样",
    ]
    has_ambiguous = any(p in question for p in ambiguous_patterns)

    is_follow_up = len(question) <= 8 and ("呢" in question or "吗" in question)

    has_implicit_link = any(p in question for p in ["后来", "然后", "之后", "那", "所以"])

    is_vague = len(question) <= 6 or any(
        question.startswith(p) for p in ["怎么", "什么意思", "为什么这样", "怎么回事"]
    )

    return has_ambiguous or is_follow_up or has_implicit_link or is_vague


def smart_rewrite(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    智能改写：判断是否需要改写，需要才调用 LLM。
    不需要则直接返回原始问题，避免浪费 LLM 调用 + 避免改写引入噪音。
    """
    if needs_rewrite(question, history):
        return rewrite_query(question, history=history)
    return question


# 示例用法
if __name__ == "__main__":
    # 单轮测试
    print(rewrite_query("那个男的后来死了吗？"))
    
    # 多轮测试
    history = [
        {"role": "user", "content": "《活着》里福贵的儿子叫什么？"},
        {"role": "assistant", "content": "有庆。"}
    ]
    print(rewrite_query("他后来怎么了？", history=history))