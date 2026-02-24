import argparse

import uvicorn

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from pipeline.embedding import Embedder
from pipeline.milvus import Milvus
from model import generate as gen, build_rag_context_block
from dotenv import load_dotenv

load_dotenv()

milvus_host = "127.0.0.1"
milvus_port = "19530"
milvus_collection = "code_chunks"
milvus_metric = "IP"
dim = 768
model_path = "krlvi/sentence-t5-base-nlpl-code_search_net"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.milvus = Milvus(host=milvus_host, port=milvus_port)
    app.state.embedder = Embedder(dim=dim, model_path=model_path)
    app.state.col = app.state.milvus.ensure_collection(
        name=milvus_collection,
        dim=dim,
        metric=milvus_metric,
    )
    yield


app = FastAPI(title="llm-coding-copilot", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prefix: str = Field(..., description="Code before cursor (required)")
    suffix: str = Field("", description="Code after cursor (optional)")
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    do_sample: Optional[bool] = Field(None, description="If omitted, inferred from temperature")
    extra_stop: List[str] = Field(default_factory=list)

    use_rag: bool = Field(False, description="Whether to retrieve internal code context")
    rag_threshold: float = Field(0.45, ge=-1.0, le=1.0, description="Min similarity score to include chunks")
    rag_top_k: int = Field(5, ge=1, le=10, description="Max chunks to retrieve")
    repo: Optional[str] = Field(None, description="Repo filter (optional)")
    branch: Optional[str] = Field(None, description="Branch filter (optional)")
    language: Optional[str] = Field(None, description="Language filter, e.g. python")
    exclude_file_path: Optional[str] = Field(None, description="Exclude current file path from retrieval")


class GenerateResponse(BaseModel):
    completion: str
    finish_reason: str  # "stop" | "length"


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> JSONResponse:
    if not req.prefix:
        raise HTTPException(status_code=400, detail="prefix must be non-empty")

    do_sample = req.do_sample if req.do_sample is not None else (req.temperature > 0.0)

    prefix = req.prefix
    suffix = req.suffix or ""

    milvus = app.state.milvus
    col = app.state.col
    embedder = app.state.embedder

    if req.use_rag:
        query_text = prefix[-2000:]

        hits = milvus.embed_and_search(
            query_text=query_text,
            col=col,
            embedder=embedder,
            threshold=req.rag_threshold,
            metric=milvus_metric,
            repo=req.repo,
            branch=req.branch,
            language=req.language,
            exclude_file_path=req.exclude_file_path,
        )

        if hits:
            ctx = build_rag_context_block(hits, language=(req.language or "python"))
            prefix = ctx + prefix

    completion, finish_reason = await gen(
        prefix,
        suffix,
        req.max_new_tokens,
        req.temperature,
        req.top_p,
        do_sample,
        req.extra_stop,
    )

    return JSONResponse({"completion": completion, "finish_reason": finish_reason})
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8005)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")