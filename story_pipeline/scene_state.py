"""
story_pipeline/scene_state.py
=============================
SharedMemory ℋ — 全局共享记忆。

所有 Agent 通过读写此对象通信，而非直接调用彼此。
借鉴 MUSE 的共享历史记忆设计。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from story_pipeline.story_config import (
    CharacterAsset,
    FrameCritique,
    FrameRecord,
    FrameSpec,
    Placement,
    StoryScript,
    VisualStyle,
)


class SharedMemory:
    """
    全局共享记忆 ℋ。

    分为两部分：
    - 不变量（Phase 0 冻结后不再修改）：script, style, characters
    - 帧级状态（逐帧更新）：frame_history, camera_history
    """

    def __init__(self) -> None:
        # ── 不变量（Phase 0 冻结后只读） ─────────────────
        self.script: Optional[StoryScript] = None
        self.style: Optional[VisualStyle] = None
        self.characters: Dict[str, CharacterAsset] = {}

        # ── 帧级状态 ──────────────────────────────────────
        self.frame_history: List[FrameRecord] = []
        self.camera_history: List[dict] = []

    # ══════════════════════════════════════════════════════
    #  Phase 0: 冻结操作
    # ══════════════════════════════════════════════════════

    def lock_script(self, script: StoryScript) -> None:
        self.script = script

    def lock_style(self, style: VisualStyle) -> None:
        self.style = style

    def lock_character(self, asset: CharacterAsset) -> None:
        self.characters[asset.character_id] = asset

    # ══════════════════════════════════════════════════════
    #  Phase 1: 帧级更新
    # ══════════════════════════════════════════════════════

    def commit_frame(self, record: FrameRecord) -> None:
        """提交一帧的完成记录。"""
        self.frame_history.append(record)
        if record.camera_params:
            self.camera_history.append(record.camera_params)

    def get_last_frame(self) -> Optional[FrameRecord]:
        return self.frame_history[-1] if self.frame_history else None

    # ══════════════════════════════════════════════════════
    #  上下文生成（给 LLM 用的摘要）
    # ══════════════════════════════════════════════════════

    def get_context_for_llm(self, window: int = 3) -> str:
        """
        生成供 LLM 使用的上下文摘要（最近 window 帧）。

        包含：
        - 全局风格名称
        - 角色列表 + 外貌摘要
        - 最近 N 帧的叙事
        """
        lines: List[str] = []

        if self.style:
            lines.append(f"[Style] {self.style.name}")

        if self.characters:
            lines.append("[Characters]")
            for cid, asset in self.characters.items():
                lines.append(f"  - {asset.name} ({cid}): {asset.appearance_prompt[:80]}...")

        recent = self.frame_history[-window:]
        if recent:
            lines.append(f"[Recent {len(recent)} frames]")
            for rec in recent:
                lines.append(f"  Frame {rec.frame_id}: {rec.narrative[:60]}...")

        return "\n".join(lines)

    def get_character(self, character_id: str) -> Optional[CharacterAsset]:
        return self.characters.get(character_id)

    # ══════════════════════════════════════════════════════
    #  持久化
    # ══════════════════════════════════════════════════════

    def save(self, output_dir: str) -> str:
        """将共享记忆状态存为 JSON。"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "shared_memory.json")
        data = {
            "style": self.style.model_dump() if self.style else None,
            "characters": {k: v.model_dump() for k, v in self.characters.items()},
            "frame_history": [f.model_dump() for f in self.frame_history],
            "camera_history": self.camera_history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
