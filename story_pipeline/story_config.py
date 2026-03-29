"""
story_pipeline/story_config.py
===============================
所有智能体间共享的 Pydantic 数据结构。
智能体之间不直接调用，而是通过这些结构体通信。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════
#  用户输入
# ══════════════════════════════════════════════════════════════════

class CharacterDef(BaseModel):
    """用户定义的角色"""
    character_id: str = Field(description="角色唯一标识")
    name: str = Field(description="角色名称")
    appearance: str = Field(description="外貌的详细文字描述")
    personality: str = Field(default="", description="性格描述")
    age: str = Field(default="", description="年龄或年龄段")
    reference_image_path: Optional[str] = Field(
        default=None, description="用户提供的角色参考图路径（可选）"
    )


class StoryInput(BaseModel):
    """用户输入（run_story.py 的入参）"""
    story_text: str = Field(description="故事描述文本（必选）")
    scene_model_path: str = Field(description="3D 场景模型路径")
    scene_iteration: int = Field(default=10000, description="场景训练迭代次数")
    characters: List[CharacterDef] = Field(
        default_factory=list,
        description="角色定义列表（可选，若为空则由编剧从故事中提取）",
    )
    character_reference_images: List[str] = Field(
        default_factory=list,
        description="角色参考图路径列表（可选）",
    )
    scene_reference_image: Optional[str] = Field(
        default=None, description="场景风格参考图路径（可选）"
    )
    style_preference: str = Field(
        default="", description="风格偏好文字描述（可选）"
    )
    num_frames: int = Field(default=6, description="期望生成的故事帧数")
    max_retries: int = Field(default=3, description="每帧最大重试次数")
    output_dir: str = Field(
        default="story_pipeline/outputs/story_run",
        description="输出目录",
    )


# ══════════════════════════════════════════════════════════════════
#  Phase 0: Pre-production 产出
# ══════════════════════════════════════════════════════════════════

class VisualStyle(BaseModel):
    """全局视觉风格锚"""
    name: str = Field(description="风格名称，如 cinematic / anime / watercolor")
    display_name: str = Field(default="", description="展示名称")
    char_prompt: str = Field(description="角色生成的风格前缀")
    scene_prompt: str = Field(description="场景描述的风格前缀")
    negative_prompt: str = Field(
        default="low quality, blurry, distorted",
        description="全局负面提示词",
    )


class CharacterAsset(BaseModel):
    """Phase 0 冻结后的角色资产"""
    character_id: str
    name: str
    appearance_prompt: str = Field(description="冻结的外貌 prompt（跨帧不变）")
    reference_image_path: Optional[str] = Field(
        default=None, description="全身参考图路径"
    )
    style_modifier: str = Field(default="", description="风格前缀")


# ══════════════════════════════════════════════════════════════════
#  Phase 0: 编剧输出 — 分镜脚本
# ══════════════════════════════════════════════════════════════════

class SceneChange(BaseModel):
    """单个场景变化"""
    object_id: str = Field(description="物品标识")
    change_type: str = Field(description="add / remove / modify")
    description: str = Field(description="变化描述")


class CharacterAction(BaseModel):
    """单帧中某角色的行为"""
    character_id: str
    action: str = Field(description="动作描述")
    position_hint: str = Field(
        default="center",
        description="位置提示：left / center / right / far-left / far-right",
    )
    emotion: str = Field(default="neutral", description="情绪")


class FrameSpec(BaseModel):
    """单帧规格（编剧输出）"""
    frame_id: int
    narrative: str = Field(description="这一帧的叙事内容")
    camera_hint: str = Field(
        default="medium shot",
        description="镜头语言提示：wide shot / medium shot / close-up / low angle / high angle / over-the-shoulder",
    )
    characters_in_frame: List[CharacterAction] = Field(default_factory=list)
    scene_changes: List[SceneChange] = Field(default_factory=list)
    atmosphere: str = Field(default="", description="氛围/光线/情绪")


class StoryScript(BaseModel):
    """编剧输出的完整分镜脚本"""
    title: str = Field(default="", description="故事标题")
    synopsis: str = Field(default="", description="故事梗概")
    characters: List[CharacterDef] = Field(description="角色定义列表")
    frames: List[FrameSpec] = Field(description="帧序列")


# ══════════════════════════════════════════════════════════════════
#  Phase 1: Production 中间产物
# ══════════════════════════════════════════════════════════════════

class Placement(BaseModel):
    """单个角色在帧中的放置方案"""
    character_id: str
    bbox: Tuple[float, float, float, float] = Field(
        description="(x1, y1, x2, y2) 归一化坐标 0~1"
    )
    depth_at_feet: float = Field(default=0.0, description="脚部深度值")
    action_prompt: str = Field(description="当前帧的动作提示词")
    scale: float = Field(default=1.0, description="相对缩放")


class LayoutPlan(BaseModel):
    """ComposerAgent 的布局规划输出"""
    valid: bool = Field(default=True)
    placements: List[Placement] = Field(default_factory=list)
    skip_reason: str = Field(default="")


# ══════════════════════════════════════════════════════════════════
#  Phase 1: Critic 产出
# ══════════════════════════════════════════════════════════════════

class Violation(BaseModel):
    """单条约束违反"""
    type: str = Field(
        description="IDENTITY_DRIFT / SPATIAL_ERROR / STYLE_MISMATCH / "
                    "INTEGRATION_ARTIFACT / ACTION_MISMATCH"
    )
    severity: str = Field(description="critical / minor")
    detail: str = Field(description="具体描述")
    fix_hint: str = Field(description="修正建议")


class FrameCritique(BaseModel):
    """CriticAgent 对一帧的验证结果"""
    passed: bool = Field(description="是否通过所有检查")
    overall_score: float = Field(default=0.0, description="总评分 0~10")
    violations: List[Violation] = Field(default_factory=list)
    revision_hints: List[str] = Field(default_factory=list)


class AssetCritique(BaseModel):
    """CriticAgent 对角色资产的验证结果"""
    passed: bool
    is_full_body: bool = Field(default=False, description="是否全身可见")
    appearance_match: float = Field(default=0.0, description="外貌匹配度 0~10")
    style_match: float = Field(default=0.0, description="风格匹配度 0~10")
    issues: List[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
#  帧结果 & 故事输出
# ══════════════════════════════════════════════════════════════════

class FrameRecord(BaseModel):
    """已完成帧的记录"""
    frame_id: int
    narrative: str = ""
    image_path: str = ""
    bg_rgb_path: str = ""
    depth_path: str = ""
    camera_params: dict = Field(default_factory=dict)
    placements: List[Placement] = Field(default_factory=list)
    critique: Optional[FrameCritique] = None


class StoryOutput(BaseModel):
    """最终输出"""
    title: str = ""
    frames: List[FrameRecord] = Field(default_factory=list)
    style: Optional[VisualStyle] = None
    characters: List[CharacterAsset] = Field(default_factory=list)
    output_dir: str = ""
