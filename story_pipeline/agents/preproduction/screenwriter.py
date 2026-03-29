"""
ScreenwriterAgent — 将用户故事扩展为结构化分镜脚本。

输入: StoryInput（故事文本 + 角色定义）
输出: StoryScript（角色列表 + 帧序列）
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from story_pipeline.agents.llm_client import LLMClient
from story_pipeline.story_config import (
    CharacterDef,
    StoryInput,
    StoryScript,
)

_SYSTEM_PROMPT = """\
你是一位专业的电影分镜编剧师。你的任务是：
将用户给定的故事描述，拆解为可执行的分镜脚本。

规则：
1. 每帧（frame）代表一个静态画面，应包含明确的视觉信息
2. 每帧必须指定哪些角色出场、各自的动作和大致位置
3. camera_hint 使用电影镜头语言：
   - wide shot（远景/全景）：建立空间关系
   - medium shot（中景）：日常对话
   - close-up（特写）：情绪表达
   - low angle（仰拍）：角色威严感
   - high angle（俯拍）：角色渺小感
   - over-the-shoulder（过肩镜头）：对话场景
4. position_hint 只用以下值：left / center / right / far-left / far-right
5. 如果用户没有提供角色定义，你需要从故事中提取角色
6. 角色的 appearance 要足够详细，能指导后续的图像生成（衣着、发型、体型等）
7. atmosphere 描述光线、天气、氛围
8. 帧数量应接近用户指定的 num_frames
"""

_USER_PROMPT_TEMPLATE = """\
请将以下故事拆解为 {num_frames} 帧的分镜脚本。

故事：
{story_text}

{character_section}

请输出 JSON 格式的分镜脚本。
"""


class ScreenwriterAgent:
    """Phase 0: 编剧智能体 — 故事 → 分镜脚本"""

    def __init__(self, client: LLMClient):
        self.client = client

    def write(self, story_input: StoryInput) -> StoryScript:
        """
        将用户输入的故事展开为结构化的分镜脚本。

        如果 story_input.characters 非空，则将已有角色定义传给 LLM；
        否则 LLM 自行从故事中提取角色。
        """
        if story_input.characters:
            char_lines = ["已定义的角色："]
            for c in story_input.characters:
                char_lines.append(
                    f"- {c.name} (ID: {c.character_id}): "
                    f"外貌={c.appearance}, 性格={c.personality}, 年龄={c.age}"
                )
            character_section = "\n".join(char_lines)
        else:
            character_section = (
                "（没有预定义角色，请从故事中提取所有角色，"
                "为每个角色生成唯一 character_id、name 和详细 appearance）"
            )

        prompt = _USER_PROMPT_TEMPLATE.format(
            num_frames=story_input.num_frames,
            story_text=story_input.story_text,
            character_section=character_section,
        )

        script = self.client.chat_json(
            prompt=prompt,
            schema=StoryScript,
            system=_SYSTEM_PROMPT,
        )

        # 如果用户已给定角色但 LLM 遗漏了，合并回去
        if story_input.characters:
            existing_ids = {c.character_id for c in script.characters}
            for c in story_input.characters:
                if c.character_id not in existing_ids:
                    script.characters.append(c)

        return script
