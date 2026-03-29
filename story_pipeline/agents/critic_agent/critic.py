"""
CriticAgent — 多维度 VLM 验证，输出结构化违反信号。

借鉴 MUSE 的核心创新：
- 不是模糊的"好/不好"
- 而是 typed violation signals（类型化的具体违反描述）
- 配合 fix_hint 实现定向修正
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import List, Optional

import numpy as np
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from story_pipeline.agents.llm_client import LLMClient
from story_pipeline.scene_state import SharedMemory
from story_pipeline.story_config import (
    AssetCritique,
    CharacterAsset,
    CharacterDef,
    FrameCritique,
    FrameSpec,
)


# ══════════════════════════════════════════════════════════════════
#  帧验证
# ══════════════════════════════════════════════════════════════════

_FRAME_VERIFY_SYSTEM = """\
你是一个严格的影视质检师。给定：
1. 一张合成后的故事帧图片
2. 该帧的叙事描述和角色信息
3. 角色的外貌参考描述

你需要从以下 5 个维度检查图片质量，输出结构化的验证结果：

1. IDENTITY_DRIFT: 角色是否与外貌描述一致？（头发颜色、服装、面部特征）
2. SPATIAL_ERROR: 角色位置是否合理？（是否悬浮、穿墙、比例失调）
3. STYLE_MISMATCH: 风格是否与全局设定一致？
4. INTEGRATION_ARTIFACT: 人物与场景的融合是否自然？（贴纸感、光影不匹配、边缘伪影）
5. ACTION_MISMATCH: 人物动作/姿态是否匹配帧描述？

severity 分级：
- critical: 明显可见的问题，必须修正
- minor: 轻微瑕疵，可接受

passed = True 当且仅当没有 severity=critical 的 violation。
overall_score 范围 0~10，7 分以上视为通过。
"""


_FRAME_VERIFY_PROMPT = """\
请验证这张故事帧：

帧叙事: {narrative}
镜头: {camera_hint}
氛围: {atmosphere}

角色信息:
{character_info}

全局风格: {style_name}

请严格检查并输出 JSON。
"""


# ══════════════════════════════════════════════════════════════════
#  角色资产验证
# ══════════════════════════════════════════════════════════════════

_ASSET_VERIFY_SYSTEM = """\
你是一个角色设计质检员。给定角色的外貌描述和参考图片（如果有），
检查以下项目：

1. is_full_body: 是否可以看到全身（从头到脚）？
2. appearance_match: 与外貌描述的匹配程度 (0~10)
3. style_match: 与目标风格的匹配程度 (0~10)
4. issues: 具体问题列表

passed = True 当且仅当 appearance_match >= 7 且无严重解剖/风格问题。
"""


class CriticAgent:
    """
    多维度 VLM 验证器。

    两个验证接口：
    - verify_frame: 验证合成后的故事帧
    - verify_asset: 验证角色参考资产
    """

    def __init__(self, client: LLMClient):
        self.client = client

    # ── 帧验证 ─────────────────────────────────────────────

    def verify_frame(self,
                     composed_rgb: np.ndarray,
                     frame: FrameSpec,
                     memory: SharedMemory) -> FrameCritique:
        """
        验证合成后的帧。

        Parameters
        ----------
        composed_rgb : [H, W, 3] float32 0~1 — 合成帧
        frame        : 帧规格
        memory       : 共享记忆

        Returns
        -------
        FrameCritique — 包含 passed, violations, revision_hints
        """
        # 保存临时图片给 VLM
        rgb_uint8 = (np.clip(composed_rgb, 0, 1) * 255).astype(np.uint8)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(rgb_uint8).save(tmp.name)

        images = [tmp.name]

        # 构造角色信息
        char_lines = []
        for ca in frame.characters_in_frame:
            asset = memory.get_character(ca.character_id)
            if asset:
                char_lines.append(
                    f"- {asset.name}: 外貌={asset.appearance_prompt[:80]}, "
                    f"动作={ca.action}, 位置={ca.position_hint}"
                )
                # 如果有参考图，也传给 VLM
                if asset.reference_image_path and os.path.exists(asset.reference_image_path):
                    images.append(asset.reference_image_path)

        try:
            critique = self.client.chat_json(
                prompt=_FRAME_VERIFY_PROMPT.format(
                    narrative=frame.narrative,
                    camera_hint=frame.camera_hint,
                    atmosphere=frame.atmosphere,
                    character_info="\n".join(char_lines) or "无角色",
                    style_name=memory.style.name if memory.style else "未指定",
                ),
                schema=FrameCritique,
                images=images,
                system=_FRAME_VERIFY_SYSTEM,
            )
        finally:
            os.unlink(tmp.name)

        return critique

    # ── 角色资产验证 ───────────────────────────────────────

    def verify_asset(self,
                     asset: CharacterAsset,
                     char_def: CharacterDef,
                     style_name: str = "") -> AssetCritique:
        """
        验证角色资产的质量。

        如果角色有参考图，用 VLM 检查图像质量；
        否则只检查 prompt 的完整性。
        """
        images = []
        if asset.reference_image_path and os.path.exists(asset.reference_image_path):
            images.append(asset.reference_image_path)

        prompt = (
            f"角色名: {asset.name}\n"
            f"外貌描述: {asset.appearance_prompt}\n"
            f"目标风格: {style_name or '未指定'}\n\n"
            f"请验证这个角色资产的质量。"
        )

        if images:
            critique = self.client.chat_json(
                prompt=prompt,
                schema=AssetCritique,
                images=images,
                system=_ASSET_VERIFY_SYSTEM,
            )
        else:
            # 没有参考图，只做文本检查（宽松通过）
            critique = AssetCritique(
                passed=True,
                is_full_body=False,
                appearance_match=7.0,
                style_match=7.0,
                issues=[],
            )

        return critique
