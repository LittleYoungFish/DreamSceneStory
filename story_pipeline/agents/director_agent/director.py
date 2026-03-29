"""
DirectorAgent — 为每帧选定最佳摄像机位。

决策流程：
1. 从 FrameSpec.camera_hint 提取镜头语言
2. CameraSelector 根据规则筛选候选相机
3. 快速渲染候选预览 → VLM 评估最佳匹配
4. 返回相机 ID + 渲染结果
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import List, Optional

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from story_pipeline.agents.director_agent.camera_selector import CameraSelector
from story_pipeline.agents.llm_client import LLMClient
from story_pipeline.scene_state import SharedMemory
from story_pipeline.story_config import FrameSpec


_SYSTEM_PROMPT = """\
你是一位电影摄影指导。给定一段帧的叙事描述和多张候选摄影角度的预览图，
选择最能表达该帧故事和情绪的摄影角度。

只输出选中的编号（如 0、1、2），不要解释。
"""


class DirectorAgent:
    """Phase 1: 为每帧选择最佳相机视角。"""

    def __init__(self, client: LLMClient, renderer, cameras_json: list):
        """
        Parameters
        ----------
        client        : LLMClient
        renderer      : SceneRenderer 实例
        cameras_json  : 完整的 cameras.json 列表
        """
        self.client = client
        self.renderer = renderer
        self.selector = CameraSelector(cameras_json)

    def select_camera(self,
                      frame: FrameSpec,
                      memory: SharedMemory,
                      num_candidates: int = 3) -> dict:
        """
        为一帧选择最佳相机。

        Returns
        -------
        dict with keys:
            cam_id   : int
            camera   : Camera 对象
            rgb      : np.ndarray [H,W,3]
            depth_raw: np.ndarray [H,W]
        """
        prev_cam_ids = [
            c.get("cam_id", 0) for c in memory.camera_history
        ]
        used_cam_ids = list(set(prev_cam_ids))

        # 获取候选相机 ID
        candidate_ids = []
        for _ in range(num_candidates):
            cam_id = self.selector.select(
                camera_hint=frame.camera_hint,
                prev_cam_ids=prev_cam_ids,
                avoid_cam_ids=used_cam_ids + candidate_ids,
                num_candidates=5,
            )
            candidate_ids.append(cam_id)

        # 去重
        candidate_ids = list(dict.fromkeys(candidate_ids))

        if len(candidate_ids) == 1:
            # 只有一个候选，直接用
            return self._render_cam(candidate_ids[0])

        # 多个候选 → 快速渲染预览 → VLM 选择
        return self._vlm_select(candidate_ids, frame)

    def _vlm_select(self, candidate_ids: list[int],
                    frame: FrameSpec) -> dict:
        """用 VLM 从多个渲染预览中选择最佳。"""
        previews = []
        temp_paths = []
        for cam_id in candidate_ids:
            result = self._render_cam(cam_id)
            # 临时保存预览图
            from PIL import Image
            rgb_uint8 = (np.clip(result["rgb"], 0, 1) * 255).astype(np.uint8)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            Image.fromarray(rgb_uint8).save(tmp.name)
            temp_paths.append(tmp.name)
            previews.append(result)

        try:
            prompt = (
                f"帧叙述: {frame.narrative}\n"
                f"镜头提示: {frame.camera_hint}\n"
                f"氛围: {frame.atmosphere}\n\n"
                f"以下是 {len(candidate_ids)} 个候选摄影角度的预览图"
                f"（编号 0 到 {len(candidate_ids)-1}）。\n"
                f"请选择最能表达这一帧故事情绪的角度编号:"
            )
            response = self.client.chat(
                prompt=prompt,
                images=temp_paths,
                system=_SYSTEM_PROMPT,
            ).strip()

            # 解析 VLM 回复中的数字
            chosen_idx = 0
            for ch in response:
                if ch.isdigit():
                    idx = int(ch)
                    if 0 <= idx < len(previews):
                        chosen_idx = idx
                    break
        finally:
            for p in temp_paths:
                os.unlink(p)

        return previews[chosen_idx]

    def _render_cam(self, cam_id: int) -> dict:
        """渲染单个相机并返回结果 + 元信息。"""
        camera = self.renderer.build_camera_from_json(cam_id)
        render_result = self.renderer.render(camera)
        return {
            "cam_id": cam_id,
            "camera": camera,
            "rgb": render_result["rgb"],
            "depth_raw": render_result["depth_raw"],
        }
