"""
ComposerAgent — 人物布局规划 + 图像合成。

替代旧的 CharacterProxy + DepthCompositor，采用 BBox + Inpainting 方案。

Plan: VLM 分析背景 → 确定角色 bbox → 深度校验
Execute: bbox → mask → style + appearance + action prompt → FLUX inpaint
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import List

import numpy as np
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from story_pipeline.agents.llm_client import LLMClient
from story_pipeline.scene_state import SharedMemory
from story_pipeline.story_config import (
    CharacterAction,
    FrameSpec,
    LayoutPlan,
    Placement,
)


# ══════════════════════════════════════════════════════════════════
#  Layout Planner（VLM 布局规划）
# ══════════════════════════════════════════════════════════════════

_LAYOUT_SYSTEM = """\
你是一个电影构图专家+计算机视觉专家。
给定一张 3D 场景渲染图和一帧的叙事描述，
你需要确定每个角色在画面中的位置（归一化 bounding box）。

规则：
1. bbox 坐标为归一化值 [0, 1]，格式 (x1, y1, x2, y2)，左上角为原点
2. 角色必须站在实体表面（地面、路面等），不能悬浮
3. bbox 高度应与角色距镜头的距离一致（远处的人更小）
4. 多角色时注意不要严重重叠
5. position_hint: left→x在0.1~0.3, center→0.35~0.65, right→0.7~0.9
6. 如果场景不适合放置角色（全是天空），设 valid=false
"""


class ComposerAgent:
    """Phase 1: 人物放置 + 图像合成"""

    def __init__(self, client: LLMClient, inpainter=None):
        """
        Parameters
        ----------
        client     : LLMClient — VLM 布局规划用
        inpainter  : FluxInpainter 实例（可选，懒加载）
        """
        self.client = client
        self.inpainter = inpainter

    # ══════════════════════════════════════════════════════════
    #  Plan: 布局规划
    # ══════════════════════════════════════════════════════════

    def plan_layout(self,
                    bg_rgb: np.ndarray,
                    depth_raw: np.ndarray,
                    frame: FrameSpec,
                    memory: SharedMemory) -> LayoutPlan:
        """
        VLM 分析背景 → 确定每个角色的 bbox。

        Parameters
        ----------
        bg_rgb    : [H, W, 3] float32 0~1
        depth_raw : [H, W] float32
        frame     : 当前帧规格
        memory    : 共享记忆

        Returns
        -------
        LayoutPlan
        """
        if not frame.characters_in_frame:
            return LayoutPlan(valid=True, placements=[], skip_reason="no characters")

        # 保存临时预览图给 VLM
        rgb_uint8 = (np.clip(bg_rgb, 0, 1) * 255).astype(np.uint8)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(rgb_uint8).save(tmp.name)

        try:
            layout = self.client.chat_json(
                prompt=self._build_layout_prompt(frame, memory),
                schema=LayoutPlan,
                images=[tmp.name],
                system=_LAYOUT_SYSTEM,
            )
        finally:
            os.unlink(tmp.name)

        # 深度校验：确保 bbox 脚部区域有合理深度
        H, W = bg_rgb.shape[:2]
        for p in layout.placements:
            foot_y = int(p.bbox[3] * H)
            foot_x = int((p.bbox[0] + p.bbox[2]) / 2 * W)
            foot_y = min(foot_y, H - 1)
            foot_x = min(foot_x, W - 1)
            p.depth_at_feet = float(depth_raw[foot_y, foot_x])

        return layout

    # ══════════════════════════════════════════════════════════
    #  Execute: 合成
    # ══════════════════════════════════════════════════════════

    def compose(self,
                bg_rgb: np.ndarray,
                layout: LayoutPlan,
                memory: SharedMemory) -> np.ndarray:
        """
        对每个 placement 执行 inpainting，逐层合成。

        按深度从远到近排序（远处先画，近处覆盖）。
        mask 外的像素完全保留 3D 渲染背景。

        Returns
        -------
        np.ndarray [H, W, 3] float32 0~1
        """
        if not layout.placements:
            return bg_rgb.copy()

        if self.inpainter is None:
            raise RuntimeError(
                "ComposerAgent.inpainter 未初始化。\n"
                "FLUX 模型需要在 dreamstory conda 环境中加载。"
            )

        H, W = bg_rgb.shape[:2]
        result = bg_rgb.copy()

        # 按深度从远到近排序（depth 大 = 远）
        sorted_placements = sorted(
            layout.placements, key=lambda p: p.depth_at_feet, reverse=True
        )

        for placement in sorted_placements:
            # 构造 mask
            mask = self._bbox_to_mask(placement, H, W)

            # 构造 prompt
            prompt = self._build_inpaint_prompt(placement, memory)

            # 执行 inpainting
            negative = ""
            if memory.style:
                negative = memory.style.negative_prompt

            result = self.inpainter.inpaint(
                bg_rgb=result,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative,
            )

        return result

    # ══════════════════════════════════════════════════════════
    #  内部工具
    # ══════════════════════════════════════════════════════════

    def _build_layout_prompt(self, frame: FrameSpec,
                              memory: SharedMemory) -> str:
        """构造送给 VLM 的布局规划 prompt。"""
        char_descs = []
        for ca in frame.characters_in_frame:
            asset = memory.get_character(ca.character_id)
            name = asset.name if asset else ca.character_id
            char_descs.append(
                f"- {name} (ID: {ca.character_id}): "
                f"动作={ca.action}, 位置提示={ca.position_hint}, "
                f"情绪={ca.emotion}"
            )

        return (
            f"帧叙事: {frame.narrative}\n"
            f"镜头: {frame.camera_hint}\n"
            f"氛围: {frame.atmosphere}\n\n"
            f"需要放置的角色:\n"
            + "\n".join(char_descs) +
            "\n\n请为每个角色确定 bbox 和 action_prompt。"
        )

    def _build_inpaint_prompt(self, placement: Placement,
                               memory: SharedMemory) -> str:
        """构造 inpainting prompt: style + appearance + action。"""
        parts: list[str] = []

        # 风格前缀（最高优先级）
        if memory.style:
            parts.append(memory.style.char_prompt)

        # 角色外貌（冻结的）
        asset = memory.get_character(placement.character_id)
        if asset:
            if asset.style_modifier:
                parts.append(asset.style_modifier)
            parts.append(asset.appearance_prompt)

        # 当前帧动作
        parts.append(placement.action_prompt)

        return ", ".join(p.strip().rstrip(",") for p in parts if p.strip())

    @staticmethod
    def _bbox_to_mask(placement: Placement,
                      H: int, W: int) -> np.ndarray:
        """将归一化 bbox 转为 bool mask。"""
        x1, y1, x2, y2 = placement.bbox
        mask = np.zeros((H, W), dtype=bool)
        px1, py1 = int(x1 * W), int(y1 * H)
        px2, py2 = int(x2 * W), int(y2 * H)
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(W, px2), min(H, py2)
        mask[py1:py2, px1:px2] = True
        return mask
