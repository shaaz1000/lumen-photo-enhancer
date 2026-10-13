import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    weights_dir: str = _env("ENHANCER_WEIGHTS", os.path.join(os.path.dirname(__file__), "..", "weights"))
    static_dir: str = _env("ENHANCER_STATIC", os.path.join(os.path.dirname(__file__), "..", "static"))
    max_upload_bytes: int = int(_env("ENHANCER_MAX_UPLOAD_MB", "25")) * 1024 * 1024
    max_input_pixels: int = int(_env("ENHANCER_MAX_INPUT_MP", "4")) * 1_000_000
    max_output_pixels: int = int(_env("ENHANCER_MAX_OUTPUT_MP", "36")) * 1_000_000
    rate_per_minute: int = int(_env("ENHANCER_RATE_PER_MINUTE", "6"))
    rate_burst: int = int(_env("ENHANCER_RATE_BURST", "3"))
    max_concurrent: int = int(_env("ENHANCER_MAX_CONCURRENT", "2"))
    tile: int = int(_env("ENHANCER_TILE", "256"))
    enable_gfpgan: bool = _env("ENHANCER_GFPGAN", "1") == "1"
    # "local" runs ONNX Runtime here; "replicate" proxies to a hosted endpoint with a server-side key
    provider: str = _env("ENHANCER_PROVIDER", "local")
    replicate_token: str = field(default_factory=lambda: os.environ.get("REPLICATE_API_TOKEN", ""), repr=False)
    replicate_version: str = _env("REPLICATE_MODEL_VERSION", "")
    trust_proxy: bool = _env("ENHANCER_TRUST_PROXY", "0") == "1"


settings = Settings()
