"""
story_pipeline/agents/llm_client.py
====================================
统一的 LLM / VLM 客户端接口。

配置
----
复制 .env.example 为 .env，填入你的密钥和模型名称：

    story_pipeline/.env
    ├── LLM_BACKEND=gemini          # 或 openai
    ├── LLM_MODEL=gemini-2.0-flash  # 模型名，留空则用下方默认值
    ├── LLM_BASE_URL=                # 自定义 API base_url（可选，留空使用官方地址）
    ├── GEMINI_API_KEY=xxx
    └── OPENAI_API_KEY=xxx

实现后端
--------
在下方 _call_api() 函数中接入你选用的 SDK，例如：

    # Gemini (pip install google-genai)
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    ...

    # OpenAI (pip install openai)
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    ...

公共接口（其余代码依赖这两个方法，不要改签名）：
    LLMClient.chat(prompt, images, system) -> str
    LLMClient.chat_json(prompt, schema, images, system) -> schema实例
"""

from __future__ import annotations

import json
import os
import re
import base64
from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# ──────────────────────────────────────────────────────────────────
#  配置加载
# ──────────────────────────────────────────────────────────────────

def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

_BACKEND  = os.environ.get("LLM_BACKEND", "gemini").lower()
_MODEL    = os.environ.get("LLM_MODEL", "")
_BASE_URL = os.environ.get("LLM_BASE_URL", "")
_API_KEY  = os.environ.get("GEMINI_API_KEY", "")


# ──────────────────────────────────────────────────────────────────
#  ★ 在这里实现你的 API 调用 ★
# ──────────────────────────────────────────────────────────────────

def _call_api(prompt: str, images: list[str], model: str, system: str, backend: str, base_url: str = "") -> str:
    if backend == "gemini":
        return _call_gemini(prompt, images, model, system, base_url)
    elif backend == "openai":
        return _call_openai(prompt, images, model, system, base_url)
    else:
        raise ValueError(f"不支持的 backend: {backend}")


def _call_gemini(prompt: str, images: list[str], model: str, system: str, base_url: str = "") -> str:
    from openai import OpenAI

    api_key = "sk-GYNoanxj28QMIObFavLZTtW5ptb3k9p19KO1hY5hp51x340o"
    base_url = "https://apinest.cn/v1"
    
    if not api_key:
        raise RuntimeError("未设置 GEMINI_API_KEY")
    if not base_url:
        raise RuntimeError("未设置 LLM_BASE_URL")

    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    content_parts = [{"type": "text", "text": prompt}]

    for img_path in images:
        img_path = os.path.abspath(img_path)
        ext = Path(img_path).suffix.lower()
        mime = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp"
        }.get(ext.lstrip("."), "image/png")

        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })

    messages.append({"role": "user", "content": content_parts})

    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content


def _call_openai(prompt: str, images: list[str], model: str, system: str, base_url: str = "") -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    content_parts = []
    for img_path in images:
        img_path = os.path.abspath(img_path)
        ext = Path(img_path).suffix.lower()
        mime = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp"
        }.get(ext.lstrip("."), "image/png")

        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })

    content_parts.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content_parts})

    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content


# ──────────────────────────────────────────────────────────────────
#  公共客户端
# ──────────────────────────────────────────────────────────────────
class LLMClient:
    _DEFAULT_MODELS = {
        "gemini": "gemini-2.0-flash",
        "openai": "gpt-4o",
    }

    def __init__(self, backend: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        self.backend = (backend or _BACKEND).lower()
        self.model = model or _MODEL or self._DEFAULT_MODELS[self.backend]
        self.base_url = base_url or _BASE_URL

    def chat(self, prompt: str, images: list[str] = None, system: str = "") -> str:
        return _call_api(prompt, images or [], self.model, system, self.backend, self.base_url)

    def chat_json(self, prompt: str, schema: Type[T], images: list[str] = None, system: str = "") -> T:
        schema_str = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        full_prompt = f"{prompt}\n\n请严格按以下 JSON Schema 输出，不要其他内容：\n{schema_str}"
        raw = self.chat(full_prompt, images, system)
        raw = re.sub(r"^```(json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw.strip())
        raw = re.sub(r',\s*}', '}', raw)
        raw = re.sub(r',\s*]', ']', raw)

        return schema.model_validate_json(raw)