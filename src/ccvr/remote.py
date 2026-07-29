from __future__ import annotations

import os
from urllib.parse import urlparse


def huggingface_endpoint() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("HF_ENDPOINT must be an absolute HTTPS endpoint")
    return endpoint
