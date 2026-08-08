# SNEPPX Model Serving Guide

## Overview

SNEPPX Serving provides a high-performance inference engine with continuous batching, paged KV cache, quantized execution, and a REST API compatible with OpenAI endpoint specifications. It includes both a Python inference server and a native C HTTP control plane.

## Python Inference Server

```python
from sneppx.serving import InferenceServer

server = InferenceServer(
    model=model,
    quantize="int4",
    max_batch_size=64,
    max_seq_len=8192,
)

server.start(port=8080)
```

### Endpoints
- `GET /v1/health`: Server health check.
- `GET /v1/models`: List loaded models.
- `POST /v1/generate`: Text generation with streaming and token counts.
- `POST /v1/generate/continuous-batch`: Continuous batching generation queue.

## C HTTP Control Plane

Built in C (`net/http/http_api.c`) with zero external dependencies (Winsock / BSD sockets):
- Bearer token authentication middleware (`http_auth.c`).
- Static file serving and route parameter capture (`{param}`).
- Demo binary: `examples/http_server_demo.c`.
