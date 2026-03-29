"""
CastingAgent — 角色资产生成与身份锚定。

为每个角色生成冻结的外貌 prompt（和可选的参考图），
确保跨帧角色一致性。
"""

from __future__ import annotations

import os
import sys
from typing import Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from story_pipeline.agents.llm_client import LLMClient
from story_pipeline.story_config import (
    AssetCritique,
    CharacterAsset,
    CharacterDef,
    VisualStyle,
)


_SYSTEM_PROMPT = """\
你是一位角色设计师。你的任务是：
将角色的文字描述扩展为详细的、可用于图像生成模型的英文外貌提示词。

规则：
1. 提示词必须是英文
2. 必须包含：性别、年龄段、发型发色、服装（从头到脚）、体型
3. 提示词应该足够具体，让不同帧生成的同一个角色看起来一致
4. 不要包含动作或姿态（那是帧级信息）
5. 不要包含背景描述
6. 只输出提示词文本，不要加其他内容
7. 生成的提示词必须符合gemini模型的输入要求，必须合规合法
"""

_REVISE_PROMPT = """\
上次生成的角色资产存在以下问题：
{issues}

请根据以上反馈修正角色外貌描述，使其更适合图像生成。
原始描述：{original_appearance}

只输出修正后的英文外貌提示词：
"""


class CastingAgent:
    """Phase 0: 角色资产构建"""

    def __init__(self, client: LLMClient):
        self.client = client

    def generate(self, char_def: CharacterDef,
                 style: VisualStyle) -> CharacterAsset:
        """
        将 CharacterDef 转换为冻结的 CharacterAsset。

        流程：
        1. LLM 将中文外貌描述扩展为英文 prompt
        2. 加上风格前缀
        3. 如果用户提供了参考图，直接使用
        """
        # 用 LLM 扩展外貌描述
        prompt = (
            f"角色名: {char_def.name}\n"
            f"外貌描述: {char_def.appearance}\n"
            f"性格: {char_def.personality}\n"
            f"年龄: {char_def.age}\n\n"
            f"请为这个角色生成详细的英文外貌提示词:"
        )
        appearance_prompt = self.client.chat(
            prompt=prompt, system=_SYSTEM_PROMPT
        ).strip()

        return CharacterAsset(
            character_id=char_def.character_id,
            name=char_def.name,
            appearance_prompt=appearance_prompt,
            reference_image_path=char_def.reference_image_path,
            style_modifier=style.char_prompt,
        )

    def revise(self, char_def: CharacterDef,
               critique: AssetCritique) -> CharacterDef:
        """根据 Critic 反馈修正角色定义。"""
        issues_text = "\n".join(f"- {issue}" for issue in critique.issues)
        revised_appearance = self.client.chat(
            prompt=_REVISE_PROMPT.format(
                issues=issues_text,
                original_appearance=char_def.appearance,
            ),
            system=_SYSTEM_PROMPT,
        ).strip()

        return CharacterDef(
            character_id=char_def.character_id,
            name=char_def.name,
            appearance=revised_appearance,
            personality=char_def.personality,
            age=char_def.age,
            reference_image_path=char_def.reference_image_path,
        )
