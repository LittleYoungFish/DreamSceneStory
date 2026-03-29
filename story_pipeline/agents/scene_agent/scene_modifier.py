"""
SceneModifier — 管理故事中场景的动态变化。

如果某帧的 scene_changes 非空（有物品增删改），
则通过 inpainting 在渲染背景上实现变化。
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict, Optional

import numpy as np
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from story_pipeline.agents.llm_client import LLMClient
from story_pipeline.scene_state import SharedMemory
from story_pipeline.story_config import FrameSpec, SceneChange


_SYSTEM_PROMPT = """\
你是一个场景修改专家。给定一张 3D 渲染的场景图和一个修改指令，
你需要确定需要修改的区域的 bounding box（归一化坐标 0~1）。

只输出 JSON，格式: {"x1": float, "y1": float, "x2": float, "y2": float, "prompt": str}
- bbox 是需要重绘的区域
- prompt 是描述修改后该区域应该呈现的样子的英文提示词
"""


class SceneModifier:
    """管理场景动态变化。通过 VLM 定位 + inpainting 实现。"""

    def __init__(self, client: LLMClient, inpainter=None):
        self.client = client
        self.inpainter = inpainter
        self._cache: Dict[str, np.ndarray] = {}

    def apply(self,
              bg_rgb: np.ndarray,
              frame: FrameSpec,
              memory: SharedMemory) -> np.ndarray:
        """
        如果帧有 scene_changes，应用变化并返回修改后的背景。
        否则原样返回。

        结果会缓存，相同变化不重复计算。
        """
        if not frame.scene_changes:
            return bg_rgb

        # 缓存 key
        cache_key = f"frame_{frame.frame_id}_" + "_".join(
            f"{sc.object_id}_{sc.change_type}" for sc in frame.scene_changes
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = bg_rgb.copy()
        for sc in frame.scene_changes:
            result = self._apply_single_change(result, sc)

        self._cache[cache_key] = result
        return result

    def _apply_single_change(self,
                              bg_rgb: np.ndarray,
                              change: SceneChange) -> np.ndarray:
        """对一个 scene_change 用 VLM 定位区域 + inpainting。"""
        if self.inpainter is None:
            # 没有 inpainter 时跳过场景修改
            return bg_rgb

        from pydantic import BaseModel, Field

        class RegionSpec(BaseModel):
            x1: float = Field(description="左上角 x (0~1)")
            y1: float = Field(description="左上角 y (0~1)")
            x2: float = Field(description="右下角 x (0~1)")
            y2: float = Field(description="右下角 y (0~1)")
            prompt: str = Field(description="修改后该区域的英文描述")

        # VLM 定位
        rgb_uint8 = (np.clip(bg_rgb, 0, 1) * 255).astype(np.uint8)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(rgb_uint8).save(tmp.name)

        try:
            region = self.client.chat_json(
                prompt=(
                    f"修改类型: {change.change_type}\n"
                    f"物品: {change.object_id}\n"
                    f"描述: {change.description}\n\n"
                    f"请确定需要修改的区域和修改后的描述。"
                ),
                schema=RegionSpec,
                images=[tmp.name],
                system=_SYSTEM_PROMPT,
            )
        finally:
            os.unlink(tmp.name)

        # 构造 mask
        H, W = bg_rgb.shape[:2]
        mask = np.zeros((H, W), dtype=bool)
        px1, py1 = int(region.x1 * W), int(region.y1 * H)
        px2, py2 = int(region.x2 * W), int(region.y2 * H)
        mask[max(0, py1):min(H, py2), max(0, px1):min(W, px2)] = True

        # Inpainting
        result = self.inpainter.inpaint(
            bg_rgb=bg_rgb,
            mask=mask,
            prompt=region.prompt,
        )
        return result
