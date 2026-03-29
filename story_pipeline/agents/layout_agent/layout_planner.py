"""
LayoutPlannerAgent
==================
根据 SceneAnalysis（VLM 场景分析结果）和相机参数，
计算出真实的 3D 角色放置坐标 + 建议体型。

核心逻辑
--------
1. 从 SceneAnalysis.character_regions 中选取置信度最高的区域
2. 取该区域底部中点作为人物脚底的像素坐标
3. 利用深度图和相机内参，将 2D 像素坐标反投影到 3D 世界坐标
4. 输出 CharacterLayout，后续直接喂给 CharacterProxy.render()
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple, Optional

import numpy as np
from pydantic import BaseModel, Field

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from story_pipeline.agents.layout_agent.scene_analyst import SceneAnalysis, BoundingBox


# ── 输出 Schema ──────────────────────────────────────────────────────────

class CharacterPlacement(BaseModel):
    """单个角色的放置方案"""
    world_position: List[float] = Field(
        description="世界坐标 [x, y, z]，人物脚底中心点"
    )
    foot_pixel: List[int] = Field(
        description="脚底在渲染图中的像素坐标 [px, py]"
    )
    height_scale: float = Field(
        default=1.0,
        description="相对于默认人物高度（0.35米世界单位）的缩放系数"
    )
    facing_direction: List[float] = Field(
        default=[0.0, 0.0, 1.0],
        description="人物朝向向量（世界坐标，XYZ），默认朝相机方向"
    )
    confidence: float = Field(description="该放置方案的置信度 0~1")
    source_bbox_index: int = Field(description="来自 SceneAnalysis.character_regions 的第几个 BoundingBox")


class LayoutPlan(BaseModel):
    """整帧的角色布局方案"""
    scene_name: str
    cam_id: int
    valid: bool = Field(description="是否存在合法的放置区域")
    placements: List[CharacterPlacement] = Field(
        description="角色放置列表，按置信度从高到低"
    )
    skip_reason: Optional[str] = Field(
        default=None,
        description="若 valid=False，说明为何跳过（如'整帧是天空，无地面'）"
    )


# ── Back-projection 工具函数 ──────────────────────────────────────────────

def unproject_pixel(
    px: int, py: int,
    depth_map: np.ndarray,
    camera,
) -> Optional[np.ndarray]:
    """
    将单个像素 (px, py) 反投影到世界 3D 坐标。

    Parameters
    ----------
    px, py      像素坐标（整数，原点左上角）
    depth_map   float32 [H, W] 深度图（相机空间距离，单位同 3DGS 场景单位）
    camera      DreamScene360 Camera 对象（含 FoVx, FoVy, R, T, image_width, image_height）

    Returns
    -------
    np.ndarray [3] 世界坐标，若深度无效则返回 None
    """
    H, W = depth_map.shape[:2]
    if px < 0 or py < 0 or px >= W or py >= H:
        return None

    depth = float(depth_map[py, px])
    if depth <= 0 or not np.isfinite(depth):
        return None

    # 归一化相机坐标
    fovx = camera.FoVx
    fovy = camera.FoVy
    tan_half_x = np.tan(fovx / 2.0)
    tan_half_y = np.tan(fovy / 2.0)

    ndc_x = (px / W - 0.5) * 2.0      # [-1, 1]
    ndc_y = (py / H - 0.5) * 2.0      # [-1, 1]

    # 注意：y 轴在图像坐标中向下为正，相机坐标中向下为负
    cam_x = ndc_x * tan_half_x * depth
    cam_y = -ndc_y * tan_half_y * depth   # 翻转 y
    cam_z = depth

    cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

    # camera.R 是 world→camera 的旋转（列优先），转置得 camera→world
    # camera.T 是 world→camera 的平移
    R = np.array(camera.R, dtype=np.float64)       # [3,3]
    T = np.array(camera.T, dtype=np.float64)       # [3]

    # camera→world：P_world = R^T @ (P_cam - T)
    world_pos = R.T @ (cam_pos - T)

    return world_pos


def pick_foot_pixel(bbox: BoundingBox) -> Tuple[int, int]:
    """
    从 BoundingBox 中选人物脚底的像素坐标。
    策略：取 bbox 底边中点（人一般站在区域底部）。
    """
    cx = (bbox.x1 + bbox.x2) // 2
    cy = bbox.y2  # 底边
    return (cx, cy)


# ── Agent ────────────────────────────────────────────────────────────────

class LayoutPlannerAgent:
    """
    将 SceneAnalysis 转换为真实的 3D 角色放置方案。
    不需要 LLM，纯几何计算。
    """

    def plan(self,
             analysis: SceneAnalysis,
             depth_map: np.ndarray,
             camera,
             scene_name: str = "unknown",
             cam_id: int = 0,
             max_placements: int = 3,
             height_scale: float = 1.0) -> LayoutPlan:
        """
        Parameters
        ----------
        analysis       SceneAnalystAgent 的输出
        depth_map      渲染深度图 np.ndarray [H, W] float32
        camera         DreamScene360 Camera 对象
        scene_name     场景名
        cam_id         相机 ID
        max_placements 最多输出多少个放置方案
        height_scale   人物高度缩放（若场景比较大，适当缩小人物）

        Returns
        -------
        LayoutPlan
        """
        if analysis.placement_impossible or not analysis.has_ground:
            return LayoutPlan(
                scene_name=scene_name,
                cam_id=cam_id,
                valid=False,
                placements=[],
                skip_reason=f"场景分析判定无法放置人物: "
                            f"placement_impossible={analysis.placement_impossible}, "
                            f"has_ground={analysis.has_ground}, "
                            f"sky_ratio={analysis.sky_ratio:.2f}"
            )

        placements: List[CharacterPlacement] = []

        for i, bbox in enumerate(analysis.character_regions[:max_placements]):
            if bbox.confidence < 0.3:
                continue

            px, py = pick_foot_pixel(bbox)

            world_pos = unproject_pixel(px, py, depth_map, camera)
            if world_pos is None:
                # 深度无效，尝试 bbox 内多个点取最近有效深度
                world_pos = self._sample_valid_depth(bbox, depth_map, camera)

            if world_pos is None:
                continue

            # 朝向：人朝相机方向（相机位置 - 人物位置，投影到 xz 平面）
            cam_world = _camera_world_position(camera)
            direction = cam_world - world_pos
            direction[1] = 0.0  # 只在 xz 平面旋转
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            else:
                direction = np.array([0.0, 0.0, 1.0])

            placements.append(CharacterPlacement(
                world_position=world_pos.tolist(),
                foot_pixel=[px, py],
                height_scale=height_scale,
                facing_direction=direction.tolist(),
                confidence=bbox.confidence,
                source_bbox_index=i,
            ))

        if not placements:
            return LayoutPlan(
                scene_name=scene_name,
                cam_id=cam_id,
                valid=False,
                placements=[],
                skip_reason="候选区域的深度值全部无效，无法反投影到 3D"
            )

        return LayoutPlan(
            scene_name=scene_name,
            cam_id=cam_id,
            valid=True,
            placements=placements,
        )

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _sample_valid_depth(self,
                            bbox: BoundingBox,
                            depth_map: np.ndarray,
                            camera) -> Optional[np.ndarray]:
        """
        在 BoundingBox 内均匀采样若干点，返回第一个深度有效的 3D 坐标。
        """
        H, W = depth_map.shape[:2]
        xs = np.linspace(bbox.x1, bbox.x2, 5, dtype=int)
        ys = np.linspace(bbox.y1, bbox.y2, 5, dtype=int)
        for y in reversed(ys):   # 从底部开始试（脚底优先）
            for x in xs:
                result = unproject_pixel(int(x), int(y), depth_map, camera)
                if result is not None:
                    return result
        return None


def _camera_world_position(camera) -> np.ndarray:
    """
    从 Camera 对象提取相机在世界坐标下的位置。
    DreamScene360 Camera: R (world→cam 旋转), T (world→cam 平移)
    => P_cam_world = -R^T @ T
    """
    R = np.array(camera.R, dtype=np.float64)
    T = np.array(camera.T, dtype=np.float64)
    return -R.T @ T
