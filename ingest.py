import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import INGEST as CFG, LANG
import pickle
import re
import shutil

def smart_chunking(text, max_chunk_chars, overlap_chars):
    """
    1. 首先按双换行（段落）分割
    2. 如果某段落超过 max_chunk_chars，再按句号/问号/感叹号进一步切分
    3. 合并小段落达到接近 max_chunk_chars，但保留段落边界
    4. 添加重叠（从前一个 chunk 末尾截取 overlap 字符）
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_chars,
        chunk_overlap=overlap_chars,
        separators=CFG["separators"][LANG],
        keep_separator=True,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks

def load_source(source: str):
    """根据文件后缀或协议选择对应的 LangChain Loader"""
    ext = os.path.splitext(source)[1].lower()
    if ext == ".pdf":
        return PyMuPDFLoader(source).load(), "pdf"
    elif ext == ".txt":
        return TextLoader(source, encoding="utf-8").load(), "txt"
    elif ext == ".md":
        return TextLoader(source, encoding="utf-8").load(), "md"
    elif ext == ".docx":
        return Docx2txtLoader(source).load(), "docx"
    elif source.startswith("http://") or source.startswith("https://"):
        return WebBaseLoader(source).load(), "web"
    else:
        raise ValueError(f"不支持的文件类型: {source}")


def export_debug_chunks(splits, file_name="debug_chunks.md"):
    with open(file_name, "w", encoding="utf-8") as f:
        f.write("# [切片报告] PDF 切片质量检查报告\n\n")
        for i, chunk in enumerate(splits):
            f.write(f"###  全局索引: {chunk.metadata.get('chunk_index', i)} | 页码: {chunk.metadata.get('page', 0) + 1}\n")
            f.write(f"**字符长度**: {len(chunk.page_content)}\n\n")
            f.write(f"```text\n{chunk.page_content}\n```\n\n")
            f.write("---\n\n")
    print(f"报告已生成：{file_name}")

def start_ingest():
    # ── 清理旧数据，避免多文档重复摄入 ──
    if os.path.exists(CFG["vector_db_dir"]):
        shutil.rmtree(CFG["vector_db_dir"])
    if os.path.exists(CFG["corpus_file"]):
        os.remove(CFG["corpus_file"])

    all_splits = []
    global_idx = 0

    for source in CFG["sources"]:
        
        docs, doc_type = load_source(source)
        print(f"加载了 {len(docs)} 个片段 ({doc_type})")

        print("正在进行语义切片...")
        for page_idx, doc in enumerate(docs):
            chunks = smart_chunking(doc.page_content,
                max_chunk_chars=CFG["chunk_size"],
                overlap_chars=CFG["overlap_chars"])
            for chunk_text in chunks:
                cleaned = re.sub(r'\s+', ' ', chunk_text)
                cleaned = re.sub(r'-\s*\d+\s*-', '', cleaned)
                cleaned = re.sub(r'http\S+', '', cleaned)
                if not cleaned.strip():
                    continue
                new_doc = type(doc)(
                    page_content=cleaned,
                    metadata={
                        "page": page_idx,
                        "doc_id": source,
                        "chunk_index": global_idx,
                        "source_type": doc_type,
                    }
                )
                all_splits.append(new_doc)
                global_idx += 1

    splits = all_splits


    export_debug_chunks(splits)

    emb_model = CFG["embedding_model"][LANG]
    print(f"正在初始化 Embedding 模型: {emb_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_model,
        encode_kwargs={"batch_size": 64}
    )

    print(f"待向量化片段: {len(splits)} 个")

    if len(splits) > 0:
        print("正在向量化并存入数据库...")
        # 分批写入，避免一次性占用过多内存
        batch_size = 100
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            if i == 0:
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=CFG["vector_db_dir"]
                )
            else:
                vectorstore.add_documents(batch)
            print(f"  进度: {min(i + batch_size, len(splits))}/{len(splits)}")
        print("构建成功")
    else:
        print("警告：该PDF无法读取有效文本")

    with open(CFG["corpus_file"], "wb") as f:
        pickle.dump(splits, f)

    print("语料索引已保存！")


if __name__ == "__main__":
    start_ingest()