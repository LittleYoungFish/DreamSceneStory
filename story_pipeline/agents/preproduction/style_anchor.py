"""
StyleAnchor — 全局视觉风格锁定。

在 Phase 0 选定风格后冻结，后续所有帧的 prompt 都加上风格前缀。
借鉴 MUSE 的 Style Preset Library。
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from story_pipeline.agents.llm_client import LLMClient
from story_pipeline.story_config import StoryScript, VisualStyle


# ══════════════════════════════════════════════════════════════════
#  内置风格库
# ══════════════════════════════════════════════════════════════════

STYLE_PRESETS: dict[str, VisualStyle] = {
    "cinematic_realistic": VisualStyle(
        name="cinematic_realistic",
        display_name="电影写实",
        char_prompt="cinematic photography, dramatic lighting, film grain, ",
        scene_prompt="cinematic wide shot, volumetric lighting, ",
        negative_prompt="cartoon, anime, painting, illustration, low quality, blurry",
    ),
    "anime_illustration": VisualStyle(
        name="anime_illustration",
        display_name="动漫插画",
        char_prompt="anime style, cel shading, vibrant colors, ",
        scene_prompt="anime background art, detailed environment, ",
        negative_prompt="realistic, photograph, 3d render, low quality",
    ),
    "watercolor": VisualStyle(
        name="watercolor",
        display_name="水彩画",
        char_prompt="watercolor illustration, soft edges, ink and wash, ",
        scene_prompt="watercolor landscape painting, wet-on-wet technique, ",
        negative_prompt="photograph, sharp edges, digital art, low quality",
    ),
    "ghibli": VisualStyle(
        name="ghibli",
        display_name="吉卜力风",
        char_prompt="Studio Ghibli style, hand-drawn animation, warm palette, ",
        scene_prompt="Ghibli-inspired background, lush environment, painterly, ",
        negative_prompt="realistic, photograph, 3d render, dark, gritty",
    ),
    "oil_painting": VisualStyle(
        name="oil_painting",
        display_name="油画风格",
        char_prompt="oil painting style, visible brushstrokes, rich texture, ",
        scene_prompt="oil painting landscape, classical composition, ",
        negative_prompt="photograph, digital art, flat colors, low quality",
    ),
}


_SYSTEM_PROMPT = """\
你是一位视觉风格顾问。根据故事的类型、情绪和用户的偏好，
从给定的风格库中选择最合适的风格名称。

只输出风格名称（如 cinematic_realistic），不要解释。
"""


class StyleAnchor:
    """Phase 0: 全局风格选定与锁定"""

    def __init__(self, client: LLMClient):
        self.client = client

    def select(self, script: StoryScript, user_pref: str = "") -> VisualStyle:
        """
        根据故事脚本 + 用户偏好选择风格。

        优先级：
        1. 用户偏好直接匹配内置风格名 → 直接使用
        2. 用户偏好非空 → LLM 从库中选择最接近的
        3. 用户偏好为空 → LLM 根据故事调性选择
        """
        # 直接匹配
        if user_pref and user_pref.lower() in STYLE_PRESETS:
            return STYLE_PRESETS[user_pref.lower()]

        # LLM 选择
        style_list = "\n".join(
            f"- {name}: {s.display_name}" for name, s in STYLE_PRESETS.items()
        )
        prompt = (
            f"故事梗概: {script.synopsis or script.title}\n"
            f"用户风格偏好: {user_pref or '无'}\n\n"
            f"可选风格:\n{style_list}\n\n"
            f"请选择一个最合适的风格名称:"
        )
        chosen = self.client.chat(prompt=prompt, system=_SYSTEM_PROMPT).strip()

        # 模糊匹配
        for name in STYLE_PRESETS:
            if name in chosen.lower():
                return STYLE_PRESETS[name]

        # 默认回退
        return STYLE_PRESETS["cinematic_realistic"]
