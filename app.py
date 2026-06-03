import streamlit as st
import json
import os
from rag_engine import generate_response_with_rewrite

HISTORY_FILE = "./chat_history.json"

def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_chat_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

st.set_page_config(page_title="本地 PDF 智能助手", page_icon="📚")
st.title("📚 我的本地知识库 (Hybrid Search 版)")
st.caption("已启用：语义向量 + BM25 关键词双引擎检索")

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.header("⚙️ 项目设置")
    use_rerank = st.toggle("精确模式 (Reranker)", value=False,
        help="关闭走 BM25（<1s），开启走 Reranker 精排（+10s，精度略高）")
    if st.button("🗑️ 清空聊天记录"):
        st.session_state.messages = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.rerun()

    st.divider()
    st.markdown(f"""
    **当前模式：**
    - 🟢 向量检索 (Vector)
    - 🟢 关键词检索 (BM25)
    - {'🟢' if use_rerank else '⚫'} Reranker 精排
    - 🟢 本地推理 (Ollama)
    """)

if prompt := st.chat_input("基于 PDF 内容提问..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🔍 正在跨维度检索并总结..."):
            try:
                history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else None
                full_response, source_docs = generate_response_with_rewrite(prompt, history, use_rerank=use_rerank)
                st.markdown(full_response)

                with st.expander("📚 查看混合检索来源"):
                    st.markdown("**本次检索结果来自以下片段（向量 + BM25 双路融合）：**")
                    for i, doc in enumerate(source_docs, 1):
                        col1, col2 = st.columns([1, 5])
                        with col1:
                            st.write(f"**[{i}]**")
                        with col2:
                            st.write(f"📄 第 {doc.metadata.get('page', '未知')} 页")
                            st.caption(doc.page_content[:300] + "...")

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_chat_history(st.session_state.messages)

            except Exception as e:
                st.error(f"发生错误啦：{e}")
                st.info("💡 提示：请确保你已经运行过 ingest.py 并生成了 bm25_corpus.pkl")