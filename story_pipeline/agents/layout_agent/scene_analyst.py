"""
SceneAnalystAgent
=================
调用 VLM（Gemini/GPT-4V）分析渲染帧，输出：
  1. 场景语义描述（地面/墙壁/天空比例）
  2. 合法的人物落脚区域（像素坐标矩形）
  3. 建议相机角色关系（人物大小、距离感）
  4. 背景描述（用于指导 Inpainting prompt）

这是整个管线中"看懂场景"的核心 Agent，没有它，Mask 位置和 Prompt 都是盲猜。
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from pydantic import BaseModel, Field

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from story_pipeline.agents.llm_client import LLMClient


# ── 输出 Schema ──────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: int = Field(description="左上角 x 像素坐标")
    y1: int = Field(description="左上角 y 像素坐标")
    x2: int = Field(description="右下角 x 像素坐标")
    y2: int = Field(description="右下角 y 像素坐标")
    confidence: float = Field(description="0~1，该区域适合放置人物的置信度")
    reason: str = Field(description="为什么认为这里适合放人物")


class SceneAnalysis(BaseModel):
    scene_type: str = Field(
        description="场景类型，例如 alley/street/room/outdoor/sky-only 等"
    )
    has_ground: bool = Field(description="画面中是否有地面/地板/路面等可供站立的区域")
    ground_description: str = Field(description="地面区域的描述，如'石板路占画面下半部分'")
    sky_ratio: float = Field(description="天空/上方背景占画面的比例 0~1")
    character_regions: List[BoundingBox] = Field(
        description="人物可自然站立的像素区域列表，按置信度从高到低排序。若无合适区域则为空列表。"
    )
    placement_impossible: bool = Field(
        description="若画面完全是天空/纯背景，无任何可放置人物的地面区域，则为 True"
    )
    scene_style_prompt: str = Field(
        description="描述背景风格的英文 prompt 片段，用于指导 Inpainting 时人物与场景的融合，例如 'dark alley with wet cobblestones, cinematic night lighting'"
    )
    recommended_character_scale: str = Field(
        description="人物在画面中的建议比例描述，例如 'full body, occupying ~30% of image height'"
    )


# ── Agent ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一个专业的电影分镜构图分析师，同时具备计算机视觉场景理解能力。

你的任务是分析给定的 3D 场景渲染图，判断：
1. 这个画面里是否有适合放置故事人物（站立角色）的地面区域
2. 如果有，精确标出人物可以自然放置的像素坐标区域
3. 如果没有（例如整帧是天空），明确指出 placement_impossible=true

关键规则：
- 角色必须站在实体表面（地面、台阶、路面等），不能悬浮在空中或天空中
- 坐标原点在左上角，x 向右，y 向下
- 若画面大部分是天空，要诚实地说 placement_impossible=true，不要强行给出坐标
- character_regions 按置信度降序排列，最多给 3 个候选区域
"""

_USER_PROMPT_TEMPLATE = """\
请分析这张 3D 场景渲染图（来自 {scene_name} 场景，相机 ID: {cam_id}）。

图像尺寸：{W}×{H} 像素。

请输出 JSON 格式的场景分析结果，重点判断：
1. 是否存在可供人物站立的地面/路面
2. 如果存在，给出最多 3 个候选放置区域的像素坐标（BoundingBox）
3. 这个场景的视觉风格（用于后续生成人物时保持一致）
"""


class SceneAnalystAgent:
    """
    分析渲染帧，输出人物放置建议。

    Parameters
    ----------
    client : LLMClient
        已配置好的 LLM 客户端（Gemini 或 OpenAI）。
    """

    def __init__(self, client: LLMClient):
        self.client = client

    def analyze(self,
                image_path: str,
                scene_name: str = "unknown",
                cam_id:     int = 0,
                W:          int = 512,
                H:          int = 512) -> SceneAnalysis:
        """
        分析渲染图，返回 SceneAnalysis。

        Parameters
        ----------
        image_path : str   渲染 RGB 图的文件路径
        scene_name : str   场景名称（提示 LLM 背景信息）
        cam_id     : int   相机 ID
        W, H       : int   图像尺寸

        Returns
        -------
        SceneAnalysis
        """
        prompt = _USER_PROMPT_TEMPLATE.format(
            scene_name=scene_name, cam_id=cam_id, W=W, H=H
        )

        analysis = self.client.chat_json(
            prompt=prompt,
            schema=SceneAnalysis,
            images=[image_path],
            system=_SYSTEM_PROMPT,
        )

        return analysis
