"""
CameraSelector — 将镜头语言提示映射为具体相机参数。

纯规则/几何计算，不调用 LLM。
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
#  镜头语言 → 参数映射表
# ══════════════════════════════════════════════════════════════════

# camera_hint → (fov_deg 范围, 距离修正系数, 高度偏移)
_SHOT_PARAMS = {
    "wide shot":           {"fov_range": (60, 75), "dist_scale": 1.5, "height_offset": 0.0},
    "medium shot":         {"fov_range": (45, 55), "dist_scale": 1.0, "height_offset": 0.0},
    "close-up":            {"fov_range": (30, 40), "dist_scale": 0.5, "height_offset": 0.1},
    "low angle":           {"fov_range": (45, 60), "dist_scale": 1.0, "height_offset": -0.3},
    "high angle":          {"fov_range": (45, 60), "dist_scale": 1.0, "height_offset": 0.4},
    "over-the-shoulder":   {"fov_range": (40, 50), "dist_scale": 0.7, "height_offset": 0.05},
}

# 默认参数
_DEFAULT_PARAMS = {"fov_range": (45, 55), "dist_scale": 1.0, "height_offset": 0.0}


class CameraSelector:
    """
    从 cameras.json 的 240 个预设中，选择最匹配镜头语言的相机。

    策略：
    1. 根据 camera_hint 确定 fov 范围
    2. 参考前序帧相机保持空间连贯
    3. 在候选中随机选择以增加多样性
    """

    def __init__(self, cameras_json: list):
        """
        Parameters
        ----------
        cameras_json : list
            从 cameras.json 加载的完整列表。
        """
        self.cameras_json = cameras_json

    def select(self,
               camera_hint: str = "medium shot",
               prev_cam_ids: list[int] = None,
               avoid_cam_ids: list[int] = None,
               num_candidates: int = 5) -> int:
        """
        选择最适合的相机 ID。

        Parameters
        ----------
        camera_hint     : 镜头语言提示
        prev_cam_ids    : 前序帧使用的相机 ID 列表（用于连贯性）
        avoid_cam_ids   : 要避免的相机 ID（已使用过的）
        num_candidates  : 返回前先筛选的候选数量

        Returns
        -------
        int : cameras.json 中的索引
        """
        params = _SHOT_PARAMS.get(camera_hint.lower(), _DEFAULT_PARAMS)
        fov_min, fov_max = params["fov_range"]
        avoid_set = set(avoid_cam_ids or [])

        # 为每个相机打分
        scores: list[Tuple[int, float]] = []
        for idx, cam in enumerate(self.cameras_json):
            if idx in avoid_set:
                continue

            # 基础 FOV 匹配分
            cam_fov = self._estimate_fov_deg(cam)
            fov_score = 1.0 - min(abs(cam_fov - (fov_min + fov_max) / 2) / 30.0, 1.0)

            # 高度偏移匹配（用 position[1] 估算）
            height = cam.get("position", [0, 0, 0])[1]
            height_offset = params["height_offset"]
            height_score = 1.0 - min(abs(height - height_offset) / 2.0, 1.0)

            # 连贯性加分：与前序相机距离不能太远
            coherence_score = 0.0
            if prev_cam_ids:
                last_id = prev_cam_ids[-1]
                if 0 <= last_id < len(self.cameras_json):
                    last_pos = self.cameras_json[last_id].get("position", [0, 0, 0])
                    cur_pos = cam.get("position", [0, 0, 0])
                    dist = sum((a - b) ** 2 for a, b in zip(last_pos, cur_pos)) ** 0.5
                    coherence_score = max(0, 1.0 - dist / 5.0)

            total = fov_score * 0.4 + height_score * 0.3 + coherence_score * 0.3
            scores.append((idx, total))

        # 取 top-N 候选，随机选一个（增加多样性）
        scores.sort(key=lambda x: x[1], reverse=True)
        candidates = scores[:num_candidates]
        if not candidates:
            return 0

        # 按分数加权随机
        weights = [s for _, s in candidates]
        total_w = sum(weights) or 1.0
        probs = [w / total_w for w in weights]
        chosen = random.choices(candidates, weights=probs, k=1)[0]
        return chosen[0]

    @staticmethod
    def _estimate_fov_deg(cam: dict) -> float:
        """从 cameras.json 条目估算水平 FOV（度）。"""
        fx = cam.get("fx", 260)
        w = cam.get("width", 512)
        fov_rad = 2 * math.atan(w / (2 * fx))
        return math.degrees(fov_rad)
