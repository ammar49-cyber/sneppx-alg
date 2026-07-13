"""CLI launcher for the SneppX inference server."""

import os
import sys
import json
import argparse
import logging


def main():
    parser = argparse.ArgumentParser(description="SneppX Inference Server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument(
        "--reload", action="store_true", help="Auto-reload on code changes (dev)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model config JSON/YAML for pre-loading",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="default",
        help="Model ID when loading from --model-config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .sneppx checkpoint to load weights from",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json file or directory",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of uvicorn workers"
    )
    parser.add_argument(
        "--auth-mode",
        type=str,
        default=None,
        choices=[None, "none", "bearer", "api_key"],
        help="Authentication mode",
    )
    parser.add_argument(
        "--api-keys",
        type=str,
        default=None,
        help="Comma-separated API keys for auth",
    )
    parser.add_argument(
        "--rate-limit-rpm",
        type=int,
        default=None,
        help="Requests per minute limit",
    )
    parser.add_argument(
        "--no-prompt-filter",
        action="store_true",
        help="Disable prompt injection detection",
    )
    parser.add_argument(
        "--no-output-verifier",
        action="store_true",
        help="Disable output verification",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Configure security
    if args.auth_mode is not None or args.rate_limit_rpm is not None:
        from .security_middleware import (
            SecurityConfig,
            AuthConfig,
            RateLimitConfig,
            PromptFilterConfig,
            OutputVerifierConfig,
        )
        from .inference_server import set_security

        auth = AuthConfig(
            mode=args.auth_mode or "none",
            api_keys=(
                [k.strip() for k in args.api_keys.split(",")]
                if args.api_keys
                else []
            ),
        )
        rate_limit = RateLimitConfig(
            enabled=(args.rate_limit_rpm or 60) > 0,
            requests_per_minute=args.rate_limit_rpm or 60,
        )
        prompt_filter = PromptFilterConfig(enabled=not args.no_prompt_filter)
        output_verifier = OutputVerifierConfig(enabled=not args.no_output_verifier)
        sec_config = SecurityConfig(
            auth=auth,
            rate_limit=rate_limit,
            prompt_filter=prompt_filter,
            output_verifier=output_verifier,
        )
        set_security(sec_config)
        logging.info(
            f"Security: auth={auth.mode}, rate_limit={rate_limit.requests_per_minute}/min, "
            f"prompt_filter={prompt_filter.enabled}, output_verifier={output_verifier.enabled}"
        )

    # Pre-load model if config provided
    if args.model_config:
        _load_model(args)

    _run_server(args)


def _load_model(args):
    from .model_zoo import build_transformer_from_config, get_model_config
    from .inference_server import register_model
    from .tokenizer import Tokenizer
    from .generation import GenerationConfig

    logging.info(f"Loading model config from {args.model_config}")
    with open(args.model_config) as f:
        if args.model_config.endswith((".yaml", ".yml")):
            import yaml

            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    if isinstance(config, dict) and "family" in config:
        model_config = get_model_config(config["family"], config.get("size", "7B"))
        model = build_transformer_from_config(model_config)
    elif isinstance(config, dict):
        model = build_transformer_from_config(config)
    else:
        logging.error("Invalid model config format")
        return

    tokenizer = None
    if args.tokenizer:
        tokenizer = Tokenizer(args.tokenizer)

    gen_config = GenerationConfig(
        max_new_tokens=config.get("max_new_tokens", 256),
        temperature=config.get("temperature", 1.0),
        top_k=config.get("top_k", 0),
        top_p=config.get("top_p", 1.0),
    )

    register_model(
        model_id=args.model_id,
        model=model,
        tokenizer=tokenizer,
        default_config=gen_config,
        meta={"config": config, "checkpoint": args.checkpoint},
    )
    logging.info(f"Model '{args.model_id}' loaded")


def _run_server(args):
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    logging.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        "SneppX_ALG.interface_bindings.inference_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
