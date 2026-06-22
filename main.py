"""
FastAPI 服务：将 RAG 管道封装为 HTTP 接口

启动：uvicorn main:app --host 0.0.0.0 --port 8000
测试：curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"question\":\"什么是自动求导？\"}"
"""

from fastapi import FastAPI
from pydantic import BaseModel

# 导入即加载 Embedding + ChromaDB + BM25 索引（只需一次，~5s）
from rag_engine import generate_response_with_rewrite

app = FastAPI(title="RAG 智能问答 API", version="1.0")


class ChatRequest(BaseModel):
    question: str
    history: list[dict] | None = None
    use_rerank: bool = False


class Source(BaseModel):
    content: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer, docs = generate_response_with_rewrite(
        req.question, req.history, use_rerank=req.use_rerank
    )
    sources = [
        Source(
            content=d.page_content[:200],
            page=d.metadata.get("page", -1),
        )
        for d in docs
    ]
    return ChatResponse(answer=answer, sources=sources)