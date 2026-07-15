"""FastAPI inference server for SneppX models — generate, chat, stream."""

import os
import time
import json
import asyncio
import logging
import uuid
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple
from contextlib import asynccontextmanager

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    FastAPI = None
    BaseModel = object

from .tensor import Tensor
from .nn import Module
from .generation import (
    GenerationConfig,
    generate,
    batch_generate,
    Streamer,
    TextStreamer,
    TokenStreamer,
)
from .tokenizer import Tokenizer
from .security_middleware import SecurityConfig, SecurityMiddleware
from .firewall import create_firewall as _create_firewall

logger = logging.getLogger(__name__)


# ===========================================================================
#  Pydantic Models
# ===========================================================================


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_new_tokens: int = Field(256, ge=1, le=4096)
    do_sample: bool = True
    temperature: float = Field(1.0, ge=0.01, le=5.0)
    top_k: int = Field(0, ge=0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.0, le=10.0)
    num_beams: int = Field(1, ge=1, le=8)
    stop_strings: Optional[List[str]] = None
    model: Optional[str] = None


class GenerateResponse(BaseModel):
    generated_text: str
    token_ids: List[int]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    created: int


class BatchGenerateRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, max_length=64)
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    model: Optional[str] = None


class BatchGenerateResponse(BaseModel):
    results: List[GenerateResponse]
    model: str
    created: int


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(1.0, ge=0.01, le=5.0)
    top_k: int = 0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "sneppx"
    meta: Dict[str, Any] = {}


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.9.5.748"
    models_loaded: int = 0
    uptime_seconds: float = 0.0


# ===========================================================================
#  Model Registry
# ===========================================================================


class ModelEntry:
    def __init__(
        self,
        model_id: str,
        model: Module,
        tokenizer: Optional[Tokenizer] = None,
        default_config: Optional[GenerationConfig] = None,
        meta: Optional[dict] = None,
    ):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.default_config = default_config or GenerationConfig()
        self.meta = meta or {}
        self.created = int(time.time())


_models: Dict[str, ModelEntry] = {}
_start_time: float = time.time()


def register_model(
    model_id: str,
    model: Module,
    tokenizer: Optional[Tokenizer] = None,
    default_config: Optional[GenerationConfig] = None,
    meta: Optional[dict] = None,
):
    _models[model_id] = ModelEntry(model_id, model, tokenizer, default_config, meta)


def get_model(model_id: Optional[str] = None) -> ModelEntry:
    if model_id is None:
        if not _models:
            raise HTTPException(503, "No models loaded")
        return list(_models.values())[0]
    if model_id not in _models:
        raise HTTPException(404, f"Model '{model_id}' not found")
    return _models[model_id]


def list_models() -> List[ModelEntry]:
    return list(_models.values())


def _count_tokens(token_ids: List[int]) -> int:
    return len(token_ids)


def _make_generation_config(req: GenerateRequest) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        num_beams=req.num_beams,
        stop_strings=req.stop_strings,
    )


def _make_chat_config(req: ChatCompletionRequest) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=req.max_tokens,
        do_sample=req.temperature > 0,
        temperature=req.temperature or 1.0,
        top_k=req.top_k,
        top_p=req.top_p,
        stop_strings=req.stop,
    )


# ===========================================================================
#  FastAPI App
# ===========================================================================


_security: Optional[SecurityMiddleware] = None


def get_security() -> Optional[SecurityMiddleware]:
    return _security


def set_security(config: Optional[SecurityConfig] = None, firewall_config: Optional[dict] = None):
    global _security
    if config is None:
        _security = SecurityMiddleware(SecurityConfig.from_env())
    else:
        _security = SecurityMiddleware(config)
    if firewall_config is not None:
        _security.firewall = _create_firewall(**firewall_config)
    elif os.environ.get("FIREWALL_CONFIG"):
        _security.firewall = _create_firewall(os.environ["FIREWALL_CONFIG"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting SneppX inference server")
    _models.clear()
    global _security
    if _security is None:
        _security = SecurityMiddleware(SecurityConfig.from_env())
    app.state.security = _security
    yield
    _models.clear()
    app.state.security = None
    logger.info("Shutting down SneppX inference server")


app = FastAPI(
    title="SneppX Inference API",
    version="0.9.5.748",

  version="0.9.5.748",
        models_loaded=len(_models),
        uptime_seconds=time.time() - _start_time,
    )


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models_endpoint():
    data = []
    for entry in _models.values():
        data.append(
            ModelInfo(
                id=entry.model_id,
                created=entry.created,
                meta=entry.meta,
            )
        )
    return ModelListResponse(data=data)


@app.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
    entry = get_model(model_id)
    return ModelInfo(
        id=entry.model_id,
        created=entry.created,
        meta=entry.meta,
    )


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest):
    filter_status, sanitized = _filter_prompt(req.prompt)
    if filter_status == "injection":
        raise HTTPException(400, "Prompt blocked by content filter")
    prompt_to_use = sanitized if filter_status == "sanitized" else req.prompt

    entry = get_model(req.model)
    gen_config = _make_generation_config(req)

    if entry.tokenizer:
        input_ids = entry.tokenizer.encode(prompt_to_use)
    else:
        input_ids = [ord(c) % entry.model.vocab_size for c in prompt_to_use]

    result = generate(
        entry.model,
        input_ids,
        generation_config=gen_config,
    )
    output_ids = result["output_ids"].flatten().tolist()
    prompt_len = len(input_ids)
    completion_len = len(output_ids) - prompt_len

    if entry.tokenizer:
        generated_text = entry.tokenizer.decode(output_ids[prompt_len:])
    else:
        generated_text = "".join(chr(t % 128) for t in output_ids[prompt_len:])

    verify_status, verified_text = _verify_output(generated_text)
    if verify_status == "blocked":
        generated_text = verified_text

    return GenerateResponse(
        generated_text=generated_text,
        token_ids=output_ids,
        prompt_tokens=prompt_len,
        completion_tokens=completion_len,
        total_tokens=len(output_ids),
        model=entry.model_id,
        created=int(time.time()),
    )


@app.post("/v1/generate/batch", response_model=BatchGenerateResponse)
async def batch_generate_endpoint(req: BatchGenerateRequest):
    sanitized_prompts = []
    for prompt in req.prompts:
        filter_status, sanitized = _filter_prompt(prompt)
        if filter_status == "injection":
            raise HTTPException(400, "Prompt blocked by content filter")
        sanitized_prompts.append(sanitized if filter_status == "sanitized" else prompt)

    entry = get_model(req.model)
    gen_config = GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )

    all_prompts = []
    for prompt in sanitized_prompts:
        if entry.tokenizer:
            ids = entry.tokenizer.encode(prompt)
        else:
            ids = [ord(c) % entry.model.vocab_size for c in prompt]
        all_prompts.append(np.array(ids, dtype=np.int64))

    result = batch_generate(entry.model, all_prompts, generation_config=gen_config)
    outputs = result["output_ids"]
    results_list = []
    for i, prompt in enumerate(req.prompts):
        output_ids = outputs[i].tolist() if outputs.ndim > 1 else outputs.tolist()
        prompt_len = len(all_prompts[i])
        completion_len = len(output_ids) - prompt_len
        if entry.tokenizer:
            generated_text = entry.tokenizer.decode(output_ids[prompt_len:])
        else:
            generated_text = "".join(chr(t % 128) for t in output_ids[prompt_len:])
        results_list.append(
            GenerateResponse(
                generated_text=generated_text,
                token_ids=output_ids,
                prompt_tokens=prompt_len,
                completion_tokens=completion_len,
                total_tokens=len(output_ids),
                model=entry.model_id,
                created=int(time.time()),
            )
        )
    return BatchGenerateResponse(
        results=results_list,
        model=entry.model_id,
        created=int(time.time()),
    )


@app.post("/v1/generate/stream")
async def generate_stream_endpoint(req: GenerateRequest):
    filter_status, sanitized = _filter_prompt(req.prompt)
    if filter_status == "injection":
        raise HTTPException(400, "Prompt blocked by content filter")
    prompt_to_use = sanitized if filter_status == "sanitized" else req.prompt

    entry = get_model(req.model)
    gen_config = _make_generation_config(req)

    if entry.tokenizer:
        input_ids = entry.tokenizer.encode(prompt_to_use)
    else:
        input_ids = [ord(c) % entry.model.vocab_size for c in prompt_to_use]

    async def event_stream() -> AsyncGenerator[str, None]:
        token_buffer = []

        class AsyncStreamer(Streamer):
            def put(self, token_id: int):
                token_buffer.append(token_id)

            def put_text(self, text: str):
                pass

            def end(self):
                pass

        streamer = AsyncStreamer()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate(
                entry.model,
                input_ids,
                generation_config=gen_config,
                streamer=streamer,
            ),
        )

        prompt_len = len(input_ids)
        for i, tid in enumerate(token_buffer):
            chunk = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": str(tid)},
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": i + 1,
                    "total_tokens": prompt_len + i + 1,
                },
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    entry = get_model(req.model)
    gen_config = _make_chat_config(req)

    if entry.tokenizer and hasattr(entry.tokenizer, "apply_chat_template"):
        prompt = entry.tokenizer.apply_chat_template(
            [m.model_dump() for m in req.messages]
        )
    else:
        prompt = (
            "\n".join(f"{m.role}: {m.content}" for m in req.messages) + "\nassistant:"
        )

    filter_status, sanitized = _filter_prompt(prompt)
    if filter_status == "injection":
        raise HTTPException(400, "Prompt blocked by content filter")
    prompt_to_use = sanitized if filter_status == "sanitized" else prompt

    if entry.tokenizer:
        input_ids = entry.tokenizer.encode(prompt_to_use)
    else:
        input_ids = [ord(c) % entry.model.vocab_size for c in prompt_to_use]

    if req.stream:
        return await _chat_stream(entry, prompt_to_use, input_ids, gen_config)

    result = generate(entry.model, input_ids, generation_config=gen_config)
    output_ids = result["output_ids"].flatten().tolist()
    prompt_len = len(input_ids)
    completion_len = len(output_ids) - prompt_len

    if entry.tokenizer:
        content = entry.tokenizer.decode(output_ids[prompt_len:])
    else:
        content = ",".join(str(t) for t in output_ids[prompt_len:])

    verify_status, verified_content = _verify_output(content)
    if verify_status == "blocked":
        content = verified_content

    resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return ChatCompletionResponse(
        id=resp_id,
        created=int(time.time()),
        model=entry.model_id,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_len,
            "total_tokens": len(output_ids),
        },
    )


async def _chat_stream(entry, prompt: str, input_ids, gen_config):
    async def event_stream() -> AsyncGenerator[str, None]:
        token_buffer = []

        class AsyncStreamer(Streamer):
            def put(self, token_id: int):
                token_buffer.append(token_id)

            def put_text(self, text: str):
                pass

            def end(self):
                pass

        streamer = AsyncStreamer()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate(
                entry.model,
                input_ids,
                generation_config=gen_config,
                streamer=streamer,
            ),
        )

        prompt_len = len(input_ids)
        for i, tid in enumerate(token_buffer):
            token_text = entry.tokenizer.decode([tid]) if entry.tokenizer else str(tid)
            chunk = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


__all__ = [
    "app",
    "register_model",
    "get_model",
    "list_models",
    "set_security",
    "get_security",
    "GenerateRequest",
    "GenerateResponse",
    "BatchGenerateRequest",
    "BatchGenerateResponse",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "HealthResponse",
    "ModelInfo",
    "ModelListResponse",
    "create_firewall",
]
