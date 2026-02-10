import argparse
import uvicorn
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from model import generate as gen
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="llm-coding-agent")


class GenerateRequest(BaseModel):
    prefix: str = Field(..., description="Code before cursor (required)")
    suffix: str = Field("", description="Code after cursor (optional)")
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    do_sample: Optional[bool] = Field(
        None, description="If omitted, inferred from temperature"
    )
    extra_stop: List[str] = Field(default_factory=list)


class GenerateResponse(BaseModel):
    completion: str
    finish_reason: str  # "stop" | "length"


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> JSONResponse:
    if not req.prefix:
        raise HTTPException(status_code=400, detail="prefix must be non-empty")

    do_sample = req.do_sample if req.do_sample is not None else (req.temperature > 0.0)

    completion, finish_reason = await gen(
        req.prefix,
        req.suffix,
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
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")