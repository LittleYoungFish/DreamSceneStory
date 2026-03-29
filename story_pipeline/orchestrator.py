"""
story_pipeline/orchestrator.py
==============================
StoryOrchestrator — 闭环编排器。

持有 SharedMemory，协调各 Agent 按三阶段执行：
  Phase 0  Pre-production（身份锚定）
  Phase 1  Production（逐帧 Plan-Execute-Verify-Revise）
  Phase 2  Post-production（全局审查）
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from story_pipeline.agents.llm_client import LLMClient
from story_pipeline.scene_state import SharedMemory
from story_pipeline.story_config import (
    CharacterDef,
    FrameRecord,
    FrameSpec,
    StoryInput,
    StoryOutput,
)

logger = logging.getLogger("story_pipeline")


class StoryOrchestrator:
    """
    中枢编排器。

    Parameters
    ----------
    story_input  : StoryInput — 用户输入
    llm_client   : LLMClient — 已配置好的 LLM 客户端
    renderer     : SceneRenderer 实例（可选，None 则跳过渲染步骤）
    inpainter    : FluxInpainter 实例（可选，None 则跳过合成步骤）
    """

    def __init__(
        self,
        story_input: StoryInput,
        llm_client: LLMClient,
        renderer=None,
        inpainter=None,
    ):
        self.input = story_input
        self.client = llm_client
        self.renderer = renderer
        self.inpainter = inpainter
        self.memory = SharedMemory()

        # 懒加载各 Agent
        self._screenwriter = None
        self._style_anchor = None
        self._casting = None
        self._director = None
        self._composer = None
        self._critic = None
        self._scene_modifier = None

    # ══════════════════════════════════════════════════════════
    #  Agent 懒加载
    # ══════════════════════════════════════════════════════════

    @property
    def screenwriter(self):
        if self._screenwriter is None:
            from story_pipeline.agents.preproduction.screenwriter import ScreenwriterAgent
            self._screenwriter = ScreenwriterAgent(self.client)
        return self._screenwriter

    @property
    def style_anchor(self):
        if self._style_anchor is None:
            from story_pipeline.agents.preproduction.style_anchor import StyleAnchor
            self._style_anchor = StyleAnchor(self.client)
        return self._style_anchor

    @property
    def casting(self):
        if self._casting is None:
            from story_pipeline.agents.preproduction.casting_agent import CastingAgent
            self._casting = CastingAgent(self.client)
        return self._casting

    @property
    def director(self):
        if self._director is None:
            from story_pipeline.agents.director_agent.director import DirectorAgent
            if self.renderer is None:
                raise RuntimeError("DirectorAgent 需要 SceneRenderer")
            self._director = DirectorAgent(
                self.client, self.renderer, self.renderer.cameras_json
            )
        return self._director

    @property
    def composer(self):
        if self._composer is None:
            from story_pipeline.agents.composer_agent.compositor import ComposerAgent
            self._composer = ComposerAgent(self.client, self.inpainter)
        return self._composer

    @property
    def critic(self):
        if self._critic is None:
            from story_pipeline.agents.critic_agent.critic import CriticAgent
            self._critic = CriticAgent(self.client)
        return self._critic

    @property
    def scene_modifier(self):
        if self._scene_modifier is None:
            from story_pipeline.agents.scene_agent.scene_modifier import SceneModifier
            self._scene_modifier = SceneModifier(self.client, self.inpainter)
        return self._scene_modifier

    # ══════════════════════════════════════════════════════════
    #  主入口
    # ══════════════════════════════════════════════════════════

    def run(self) -> StoryOutput:
        """
        执行完整的三阶段故事生成管线。

        Returns
        -------
        StoryOutput — 包含所有帧记录 + 元信息
        """
        os.makedirs(self.input.output_dir, exist_ok=True)

        logger.info("═══ Phase 0: Pre-production ═══")
        self._phase0_preproduction()

        logger.info("═══ Phase 1: Production ═══")
        self._phase1_production()

        logger.info("═══ Phase 2: Post-production ═══")
        self._phase2_postproduction()

        # 保存共享记忆
        self.memory.save(self.input.output_dir)

        output = StoryOutput(
            title=self.memory.script.title if self.memory.script else "",
            frames=list(self.memory.frame_history),
            style=self.memory.style,
            characters=list(self.memory.characters.values()),
            output_dir=self.input.output_dir,
        )
        logger.info(f"故事生成完成，共 {len(output.frames)} 帧，输出目录: {output.output_dir}")
        return output

    # ══════════════════════════════════════════════════════════
    #  Phase 0: Pre-production
    # ══════════════════════════════════════════════════════════

    def _phase0_preproduction(self) -> None:
        """
        0.1 故事 → 分镜脚本
        0.2 风格锚定
        0.3 角色资产生成 + 验证循环
        """
        # 0.1 编剧
        logger.info("[Phase 0.1] 编剧：故事 → 分镜脚本")
        script = self.screenwriter.write(self.input)
        self.memory.lock_script(script)
        logger.info(f"  脚本: {script.title}, {len(script.frames)} 帧, "
                    f"{len(script.characters)} 角色")

        # 0.2 风格锚定
        logger.info("[Phase 0.2] 风格锚定")
        style = self.style_anchor.select(script, self.input.style_preference)
        self.memory.lock_style(style)
        logger.info(f"  选定风格: {style.name} ({style.display_name})")

        # 0.3 角色资产
        logger.info("[Phase 0.3] 角色资产生成")
        for char_def in script.characters:
            asset = self._generate_character_asset(char_def)
            self.memory.lock_character(asset)
            logger.info(f"  ✓ {asset.name}: {asset.appearance_prompt[:60]}...")

    def _generate_character_asset(self, char_def: CharacterDef):
        """生成角色资产，附带验证循环。"""
        style = self.memory.style
        for attempt in range(self.input.max_retries):
            asset = self.casting.generate(char_def, style)
            critique = self.critic.verify_asset(
                asset, char_def,
                style_name=style.name if style else "",
            )
            if critique.passed:
                return asset
            logger.info(f"  角色 {char_def.name} 第{attempt+1}次验证未通过: "
                       f"{critique.issues}")
            char_def = self.casting.revise(char_def, critique)

        # 超过重试次数，使用最后一版
        return asset

    # ══════════════════════════════════════════════════════════
    #  Phase 1: Production（逐帧生成）
    # ══════════════════════════════════════════════════════════

    def _phase1_production(self) -> None:
        """逐帧执行 Plan-Execute-Verify-Revise 循环。"""
        if not self.memory.script:
            raise RuntimeError("Phase 0 未完成，没有脚本")

        for frame in self.memory.script.frames:
            logger.info(f"[Phase 1] 生成帧 {frame.frame_id}: {frame.narrative[:40]}...")
            record = self._produce_frame(frame)
            self.memory.commit_frame(record)
            logger.info(f"  ✓ 帧 {frame.frame_id} 完成")

    def _produce_frame(self, frame: FrameSpec) -> FrameRecord:
        """单帧的 Plan-Execute-Verify-Revise 循环。"""
        best_record = None
        best_score = -1.0

        for attempt in range(self.input.max_retries):
            record = self._execute_frame(frame, attempt)

            # 如果没有渲染器或合成器，跳过验证
            if not record.image_path:
                return record

            # Verify
            composed_rgb = np.array(Image.open(record.image_path)).astype(np.float32) / 255.0
            critique = self.critic.verify_frame(composed_rgb, frame, self.memory)
            record.critique = critique

            if critique.passed:
                return record

            if critique.overall_score > best_score:
                best_score = critique.overall_score
                best_record = record

            logger.info(f"  帧 {frame.frame_id} 第{attempt+1}次验证未通过 "
                       f"(score={critique.overall_score:.1f})")
            for v in critique.violations:
                if v.severity == "critical":
                    logger.info(f"    [{v.type}] {v.detail}")

        # 超过重试次数，返回最优结果
        return best_record or record

    def _execute_frame(self, frame: FrameSpec, attempt: int) -> FrameRecord:
        """执行单帧的 Plan + Execute 阶段。"""
        frame_dir = os.path.join(self.input.output_dir, f"frame_{frame.frame_id:04d}")
        os.makedirs(frame_dir, exist_ok=True)

        record = FrameRecord(
            frame_id=frame.frame_id,
            narrative=frame.narrative,
        )

        if self.renderer is None:
            # 无渲染器：仅生成脚本和资产，不渲染图像
            logger.info("  跳过渲染（无 SceneRenderer）")
            return record

        # ── Plan: 选择相机 + 渲染背景 ──
        cam_result = self.director.select_camera(frame, self.memory)
        bg_rgb = cam_result["rgb"]
        depth_raw = cam_result["depth_raw"]
        cam_id = cam_result["cam_id"]

        record.camera_params = {"cam_id": cam_id}

        # 保存背景
        bg_path = os.path.join(frame_dir, f"bg_rgb_attempt{attempt}.png")
        bg_uint8 = (np.clip(bg_rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(bg_uint8).save(bg_path)
        record.bg_rgb_path = bg_path

        # 保存深度
        depth_path = os.path.join(frame_dir, f"depth_attempt{attempt}.npy")
        np.save(depth_path, depth_raw)
        record.depth_path = depth_path

        # ── Plan: 场景变化 ──
        bg_rgb = self.scene_modifier.apply(bg_rgb, frame, self.memory)

        # ── Plan: 角色布局 ──
        layout = self.composer.plan_layout(bg_rgb, depth_raw, frame, self.memory)
        record.placements = layout.placements

        if not layout.valid or not layout.placements:
            # 无需放置角色，直接保存背景
            record.image_path = bg_path
            return record

        # ── Execute: 合成 ──
        if self.inpainter is None:
            logger.info("  跳过合成（无 FluxInpainter）")
            record.image_path = bg_path
            return record

        composed = self.composer.compose(bg_rgb, layout, self.memory)
        composed_path = os.path.join(frame_dir, f"composed_attempt{attempt}.png")
        composed_uint8 = (np.clip(composed, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(composed_uint8).save(composed_path)
        record.image_path = composed_path

        return record

    # ══════════════════════════════════════════════════════════
    #  Phase 2: Post-production
    # ══════════════════════════════════════════════════════════

    def _phase2_postproduction(self) -> None:
        """全局一致性审查（简化版）。"""
        frames = self.memory.frame_history
        if len(frames) <= 1:
            return

        # 收集所有有图像的帧
        image_paths = [f.image_path for f in frames if f.image_path]
        if len(image_paths) < 2:
            return

        logger.info(f"[Phase 2] 全局一致性审查 ({len(image_paths)} 帧)")

        # 用 VLM 做整体风格一致性评估
        prompt = (
            f"这是一个故事的 {len(image_paths)} 帧场景。\n"
            f"故事标题: {self.memory.script.title if self.memory.script else '未知'}\n"
            f"目标风格: {self.memory.style.name if self.memory.style else '未指定'}\n\n"
            f"请评估这些帧之间的视觉风格一致性和叙事流畅性。\n"
            f"给出 1-10 分的总体评分和简要评语。"
        )

        try:
            assessment = self.client.chat(
                prompt=prompt,
                images=image_paths[:6],  # VLM 一次最多看 6 张
                system="你是一位影视作品质检师，负责评估故事帧的整体一致性。",
            )
            logger.info(f"  全局评估: {assessment[:200]}")
        except Exception as e:
            logger.warning(f"  全局评估失败: {e}")
