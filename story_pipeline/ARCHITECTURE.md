# 场景一致的多智能体故事生成系统 — 架构方案 v2

> 借鉴 MUSE (arXiv:2602.03028) 的闭环约束执行思想，结合 3D 场景作为世界锚点的独特优势。

## 一、总体思路

### 1.1 核心理念

**3D 场景 = 不可变的世界锚点**。所有故事帧从同一个 3DGS 模型渲染背景，物理空间天然一致，**这是我们相较于 MUSE 等纯 2D 方案的结构性优势**。

**从 MUSE 借鉴的关键思想**：
1. **闭环约束执行 (Plan → Execute → Verify → Revise)**：不是 feed-forward 一次出结果，而是生成后用 VLM 验证、发现违反约束则定向修正
2. **三阶段制片流程**：Pre-production（身份锚定）→ Production（空间合成）→ Post-production（时序一致）
3. **显式可执行约束**：叙事意图不只是自然语言 prompt，而是结构化的、可机器验证的控制信号
4. **全局共享记忆 ℋ**：跨帧持久存储角色身份、场景状态、已接受的布局等

### 1.2 与 MUSE 的关键差异

| 维度 | MUSE | 我们的系统 |
|------|------|-----------|
| 场景来源 | 文生图/文生视频，场景每帧重新生成 | **3D 场景渲染**，背景物理一致 |
| 场景一致性 | 靠 prompt + 风格锚定（软约束） | **3DGS 渲染保证（硬约束）** |
| 输出形式 | 连续视频 | 离散故事帧（关键帧） |
| 人物注入 | 文生图直接生成含人物的画面 | **背景渲染 + 人物 inpainting 注入** |
| 深度信息 | 无显式深度 | **有精确深度图，支持遮挡推理** |
| 音频 | 有 VTS 语音合成 | 暂不涉及 |

### 1.3 为什么放弃骨骼方案

现有 `CharacterProxy` 用 T-pose 骨骼 + pyrender：
- 只有僵硬的 T-pose，无法表达动作
- 渲染质量粗糙，仅作为 mask 参考
- 增加了不必要的复杂度

**新方案：BBox + Depth-aware Inpainting**（借鉴 MUSE 的 Layout-Aware 思路）：
1. VLM 分析场景 → 输出人物应在的 **bounding box 区域**
2. 根据 depth map 估算该区域的合理人物缩放
3. 用 bounding box 构造 inpainting mask
4. FLUX Fill 在 mask 区域生成人物（有场景背景上下文，自然融合）
5. **不破坏 mask 外的场景** → 场景一致性自然保持

---

## 二、系统架构总览

```
用户故事输入 (自然语言 + 角色定义 + 风格偏好)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                 StoryOrchestrator (闭环编排器)                  │
│    Plan → Execute → Verify → Revise 循环                      │
│    持有全局共享记忆 ℋ (SceneState)                              │
└────┬──────────┬──────────┬──────────┬──────────┬─────────────┘
     │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼
┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐
│Pre-prod ││Director ││Composer ││Critic   ││Scene    │
│Agent    ││Agent    ││Agent    ││Agent    ││Agent    │
│角色资产  ││导演/视角 ││合成/放置 ││验证/修正 ││场景状态  │
│+风格锚定 ││         ││         ││(VLM)    ││         │
└─────────┘└─────────┘└─────────┘└─────────┘└─────────┘
                         │
              ┌──────────┴──────────┐
              │    SceneRenderer     │
              │  (3DGS 渲染，可替换)  │
              └─────────────────────┘
```

---

## 三、三阶段制片流程

### Phase 0: Pre-production（预制作 — 身份锚定）
> 在任何帧生成之前，先锁定全局不变量

1. **故事拆帧**：LLM 将用户故事分解为帧序列 `List[FrameSpec]`
2. **角色资产生成**：为每个角色生成 **全身参考图**（白背景，正面+侧面）
3. **风格锚定**：确定全局视觉风格（style_modifier 字符串）
4. **VLM 验证**：检查角色资产是否符合描述、角色间风格是否统一
5. **冻结**：通过验证的资产写入共享记忆 ℋ，后续不再修改

### Phase 1: Production（制作 — 逐帧生成）
> 逐帧执行，每帧走 Plan-Execute-Verify-Revise 循环

1. **Plan**：导演选视角 → 渲染 3D 背景 → VLM 分析场景 → 规划人物 bbox
2. **Execute**：构造 mask → 组装 prompt → FLUX inpainting 注入人物
3. **Verify**：VLM 检查生成结果（角色身份、空间合理性、风格一致）
4. **Revise**：若违反约束，调整 prompt/mask/参数后重试（最多 N 次）

### Phase 2: Post-production（后制 — 全局审查）
> 所有帧生成后，全局一致性检查

1. 相邻帧的角色外貌 CLIP 相似度检查
2. 场景变化逻辑一致性检查
3. 输出最终故事帧序列

---

## 四、目录结构

```
story_pipeline/
├── ARCHITECTURE.md              # 本文档
├── orchestrator.py              # StoryOrchestrator — 闭环编排器
├── scene_state.py               # SharedMemory — 全局共享记忆 ℋ
├── story_config.py              # Pydantic 数据结构定义
├── run_story.py                 # CLI 入口
├── .env                         # API 密钥配置
│
├── agents/
│   ├── llm_client.py            # LLM/VLM 统一客户端 (Gemini nanobanana)
│   │
│   ├── preproduction/           # Phase 0: 预制作
│   │   ├── __init__.py
│   │   ├── screenwriter.py      # ScreenwriterAgent: 故事→分镜脚本
│   │   ├── casting_agent.py     # CastingAgent: 角色资产生成+验证
│   │   └── style_anchor.py      # StyleAnchor: 全局风格锁定
│   │
│   ├── director_agent/          # Phase 1: 导演
│   │   ├── __init__.py
│   │   ├── director.py          # DirectorAgent: 视角选择
│   │   └── camera_selector.py   # CameraSelector: 相机参数映射
│   │
│   ├── composer_agent/          # Phase 1: 合成 (替代旧的 character_agent)
│   │   ├── __init__.py
│   │   ├── layout_planner.py    # LayoutPlanner: VLM→bbox→mask
│   │   └── compositor.py        # Compositor: inpainting 合成
│   │
│   ├── critic_agent/            # Phase 1: 验证 (MUSE 的核心创新)
│   │   ├── __init__.py
│   │   └── critic.py            # CriticAgent: 多维度 VLM 验证
│   │
│   ├── scene_agent/             # 场景状态管理
│   │   ├── __init__.py
│   │   └── scene_modifier.py    # SceneModifier: 动态场景变化
│   │
│   └── layout_agent/            # 底层渲染能力 (保留)
│       ├── scene_renderer.py    # SceneRenderer [保留不动]
│       ├── scene_analyst.py     # SceneAnalystAgent [保留不动]
│       └── layout_planner.py    # LayoutPlannerAgent [保留不动]
│
├── inpainting_agent/            # 图像合成 (保留)
│   └── flux_inpainter.py        # FluxInpainter [保留不动]
│
└── outputs/                     # 运行输出
```

---

## 五、各智能体详细设计

### 5.1 StoryOrchestrator（闭环编排器）

```python
class StoryOrchestrator:
    """
    中枢编排器 — 借鉴 MUSE 的闭环编排思想。
    持有共享记忆 ℋ，协调各 Agent 按三阶段执行。
    """
    
    def run(self, story_input: StoryInput) -> StoryOutput:
        # ═══ Phase 0: Pre-production ═══
        # 0.1 故事 → 分镜脚本
        script = self.screenwriter.write(story_input)
        
        # 0.2 风格锚定 (冻结后不变)
        style = self.style_anchor.select(script, story_input.style_preference)
        self.memory.lock_style(style)
        
        # 0.3 角色资产生成 + 验证循环
        for char_def in script.characters:
            for attempt in range(MAX_RETRIES):
                asset = self.casting.generate(char_def, style)
                critique = self.critic.verify_asset(asset, char_def)
                if critique.passed:
                    self.memory.lock_character(asset)
                    break
                # 定向修正
                char_def = self.casting.revise(char_def, critique)
        
        # ═══ Phase 1: Production（逐帧） ═══
        results = []
        for frame in script.frames:
            result = self._produce_frame(frame)
            results.append(result)
            self.memory.commit_frame(result)
        
        # ═══ Phase 2: Post-production ═══
        self._global_consistency_check(results)
        
        return StoryOutput(frames=results)
    
    def _produce_frame(self, frame: FrameSpec) -> FrameResult:
        """单帧的 Plan-Execute-Verify-Revise 循环"""
        for attempt in range(MAX_RETRIES):
            # Plan
            camera = self.director.select_camera(frame, self.memory)
            bg = self.renderer.render(camera)
            bg = self.scene_modifier.apply(bg, frame, self.memory)
            layout = self.composer.plan_layout(bg, frame, self.memory)
            
            # Execute
            composed = self.composer.compose(bg, layout, self.memory)
            
            # Verify
            critique = self.critic.verify_frame(composed, frame, self.memory)
            if critique.passed:
                return FrameResult(image=composed, ...)
            
            # Revise (定向修正，不是整体重来)
            frame = self._apply_revision(frame, critique)
        
        # 超过重试次数，返回最优结果
        return best_result
```

### 5.2 ScreenwriterAgent（编剧智能体）

```python
class ScreenwriterAgent:
    """Phase 0: 将用户故事扩展为结构化分镜脚本"""
    
    def write(self, story_input: StoryInput) -> StoryScript:
        """
        输入: 用户的简短故事描述
        输出: 结构化脚本，包含：
        - characters: 角色定义列表（外貌、性格、年龄）
        - frames: 帧序列，每帧包含：
            - narrative: 叙事内容
            - camera_hint: 镜头语言（远景/近景/仰拍等）
            - characters_in_frame: 出场角色 + 动作 + 位置提示
            - scene_changes: 场景物品变化
            - atmosphere: 氛围/光线/情绪
        
        LLM提示要点：
        - 以电影分镜师的思维拆解
        - 每帧必须有明确的空间位置提示
        - 角色描述要足够详细以供后续生成
        """
```

### 5.3 CastingAgent（角色选角智能体）

> **核心创新：用 FLUX 生成角色参考图，然后 inpainting 时将参考图作为视觉条件**

```python
class CastingAgent:
    """Phase 0: 角色资产生成与身份锚定"""
    
    def generate(self, char_def: CharacterDef, style: StyleAnchor) -> CharacterAsset:
        """
        生成流程：
        1. 组装 prompt = style.char_prompt + char_def.appearance
        2. 用 FLUX 生成全身正面参考图（白背景）
        3. 去背景（RMBG 或简单阈值）
        4. 保存为角色锚定资产
        
        资产内容：
        - reference_image: 全身正面参考图 (PNG, 白背景)
        - appearance_prompt: 冻结的外貌提示词
        - style_modifier: 风格前缀
        """
    
    def revise(self, char_def, critique) -> CharacterDef:
        """根据 Critic 反馈修正角色定义"""
```

**角色一致性策略（借鉴 MUSE）**：
- 参考图在 Phase 0 生成后**冻结**，写入共享记忆
- 每帧生成时，appearance_prompt 完全复用（仅动作/姿态变化）
- style_modifier 作为 prompt 最前缀注入，优先级最高
- 参考图可用于 IP-Adapter 条件注入（如可用）

### 5.4 StyleAnchor（风格锚定）

```python
class VisualStyle(BaseModel):
    """视觉风格定义（借鉴 MUSE 的 style preset）"""
    name: str                # 如 "watercolor", "anime", "cinematic"
    char_prompt: str         # 角色生成的风格前缀
    scene_prompt: str        # 场景描述的风格前缀
    negative_prompt: str     # 全局负面提示词

# 内置风格库
STYLES = {
    "cinematic_realistic": VisualStyle(
        name="cinematic_realistic",
        char_prompt="cinematic photography, dramatic lighting, film grain, ",
        scene_prompt="cinematic wide shot, volumetric lighting, ",
        negative_prompt="cartoon, anime, painting, illustration, low quality",
    ),
    "anime_illustration": VisualStyle(
        name="anime_illustration",
        char_prompt="anime style, cel shading, vibrant colors, ",
        scene_prompt="anime background art, detailed environment, ",
        negative_prompt="realistic, photograph, 3d render",
    ),
    "watercolor": VisualStyle(
        name="watercolor",
        char_prompt="watercolor illustration, soft edges, ink and wash, ",
        scene_prompt="watercolor landscape painting, wet-on-wet technique, ",
        negative_prompt="photograph, sharp edges, digital art",
    ),
}

class StyleAnchor:
    def select(self, script: StoryScript, user_pref: str) -> VisualStyle:
        """LLM 根据故事调性 + 用户偏好选择/调整风格"""
```

### 5.5 DirectorAgent（导演智能体）

```python
class DirectorAgent:
    """Phase 1: 为每帧选定摄像机位"""
    
    def select_camera(self, frame: FrameSpec, memory: SharedMemory) -> Camera:
        """
        决策流程：
        1. 从 frame.camera_hint 提取镜头语言
           （"远景" → 视野宽, "特写" → 视野窄 + 靠近, "仰拍" → 低角度）
        2. 生成候选相机参数（多个 look_at 方向 + fov 组合）
        3. 快速渲染 3-5 个候选预览图
        4. VLM 评估哪个最匹配帧描述
        5. 返回最佳相机
        
        约束：
        - 同一场景区域的帧应使用相近视角（空间连贯）
        - 参考 memory 中前序帧的相机位（避免突兀跳转）
        """
```

### 5.6 ComposerAgent（合成智能体）

> **替代旧的 CharacterProxy + DepthCompositor，采用 BBox + Inpainting 方案**

```python
class ComposerAgent:
    """Phase 1: 人物放置 + 图像合成"""
    
    def plan_layout(self, bg_result: RenderResult, frame: FrameSpec,
                     memory: SharedMemory) -> LayoutPlan:
        """
        Step 1: VLM 分析背景图 → 识别可放置区域
        Step 2: 根据 frame 中的角色位置提示 → 确定 bbox
        Step 3: 深度图校验 → bbox 区域的深度是否合理（地面/可站立）
        Step 4: 根据深度估算人物缩放比例
        
        输出 LayoutPlan:
        - placements: List[Placement]
            - bbox: BoundingBox (x1, y1, x2, y2 归一化坐标)
            - depth_at_feet: float (脚部深度，用于缩放)
            - character_id: str
            - action_prompt: str
        """
    
    def compose(self, bg_result: RenderResult, layout: LayoutPlan,
                 memory: SharedMemory) -> np.ndarray:
        """
        对每个 placement：
        1. 从 bbox 构造 inpainting mask（矩形 + 膨胀）
        2. 组装 prompt:
           = style.char_prompt + character.appearance_prompt + action_prompt
        3. 调用 FLUX Fill inpainting
        4. 多角色时，按深度从远到近依次 inpaint（远处先画，近处覆盖）
        
        关键：mask 外的像素完全保留 3D 渲染的背景 → 场景一致性
        """
```

**为什么 BBox + Inpainting 优于骨骼方案**：
1. **更自然**：FLUX 直接理解"一个人站在巷子里"，比骨骼 + 贴图自然得多
2. **更简单**：无需 pyrender、trimesh 等 3D 依赖
3. **保留场景**：mask 外区域完全不动，3D 渲染的背景 100% 保持
4. **深度感知**：通过 bbox 大小 + 深度图，自动处理远近缩放
5. **动作灵活**：任何动作都可以通过文本 prompt 描述

### 5.7 CriticAgent（验证智能体）

> **MUSE 的核心贡献 — 闭环约束执行的关键**

```python
class CriticAgent:
    """
    多维度 VLM 验证，输出结构化的违反信号。
    借鉴 MUSE: 不是模糊的"好/不好"，而是具体的 typed violation signals。
    """
    
    def verify_asset(self, asset: CharacterAsset, 
                      char_def: CharacterDef) -> AssetCritique:
        """Phase 0: 验证角色资产
        检查项：
        - 是否全身可见（头到脚）
        - 是否匹配外貌描述
        - 解剖结构是否正确（手指数量、面部结构）
        - 风格是否匹配全局设定
        """
    
    def verify_frame(self, composed: np.ndarray, frame: FrameSpec,
                      memory: SharedMemory) -> FrameCritique:
        """Phase 1: 验证生成帧
        检查项：
        1. identity_check: 角色是否像参考图中的人？
        2. spatial_check: 人物位置是否合理？（非浮空/穿墙）
        3. style_check: 风格是否与其他帧一致？
        4. action_check: 人物动作是否匹配帧描述？
        5. integration_check: 人物与场景融合是否自然？（无贴纸感）
        
        输出 FrameCritique:
        - passed: bool
        - violations: List[Violation]  # 具体违反项
        - revision_hints: List[str]    # 修正建议
        """

class Violation(BaseModel):
    type: str      # "IDENTITY_DRIFT" | "SPATIAL_ERROR" | "STYLE_MISMATCH" | 
                   # "INTEGRATION_ARTIFACT" | "ACTION_MISMATCH"
    severity: str  # "critical" | "minor"
    detail: str    # 具体描述
    fix_hint: str  # 修正建议
```

### 5.8 SceneModifier（场景动态管理）

```python
class SceneModifier:
    """管理故事中场景的动态变化"""
    
    def apply(self, bg: RenderResult, frame: FrameSpec, 
              memory: SharedMemory) -> RenderResult:
        """
        如果当前帧有场景变化（物品增/删/改）：
        1. 从 memory 获取变化记录
        2. VLM 定位变化物品在当前视角的位置
        3. 用 inpainting 实现变化
        4. 缓存结果到 memory（相近视角可复用）
        
        注意：变化后的背景替换原始 3D 渲染结果，
        后续人物注入在修改后的背景上进行
        """
```

---

## 六、共享记忆 ℋ（SharedMemory）

```python
class SharedMemory:
    """
    全局共享记忆（借鉴 MUSE 的 ℋ）。
    所有 Agent 通过读写此对象通信，而非直接调用彼此。
    """
    
    # ── 不变量（Phase 0 冻结后不再修改） ──
    script: StoryScript                        # 分镜脚本
    style: VisualStyle                         # 全局风格锚
    characters: Dict[str, CharacterAsset]      # 角色资产（含参考图）
    
    # ── 帧级状态（逐帧更新） ──
    frame_history: List[FrameRecord]           # 已完成帧记录
    scene_objects: Dict[str, SceneObject]      # 场景物品状态
    camera_history: List[CameraParams]         # 相机历史（用于连贯性）
    
    # ── 缓存 ──
    scene_modification_cache: Dict[str, np.ndarray]  # 视角→修改后背景
    
    def get_context_for_llm(self, window: int = 3) -> str:
        """
        生成供 LLM 使用的上下文摘要（只看最近 window 帧）。
        借鉴 MUSE 的 context sliding window:
        - 当前场景状态
        - 角色当前状态（位置、情绪）
        - 最近 N 帧的叙事摘要
        """
```

---

## 七、数据流：单帧的 Plan-Execute-Verify-Revise 循环

```
┌─────────────── Plan ────────────────┐
│                                      │
│  DirectorAgent.select_camera(frame)  │
│  └→ Camera + 渲染 RGB+Depth          │
│                                      │
│  SceneModifier.apply(bg, changes)   │
│  └→ 修改后背景（如有变化）            │
│                                      │
│  ComposerAgent.plan_layout(bg)      │
│  └→ LayoutPlan (bbox + 缩放)         │
│                                      │
├─────────────── Execute ─────────────┤
│                                      │
│  ComposerAgent.compose(bg, layout)  │
│  └→ style_prompt + char_prompt       │
│     + action_prompt → FLUX inpaint   │
│  └→ 合成图 (人物 + 场景)             │
│                                      │
├─────────────── Verify ──────────────┤
│                                      │
│  CriticAgent.verify_frame(composed) │
│  └→ FrameCritique                    │
│     ├ PASSED → commit to memory      │
│     └ FAILED → typed violations      │
│                                      │
├─────────────── Revise ──────────────┤
│                                      │
│  根据 violation 类型定向修正:         │
│  IDENTITY_DRIFT →强化 appearance prompt│
│  SPATIAL_ERROR → 调整 bbox 位置/大小   │
│  STYLE_MISMATCH → 增加 style 权重     │
│  INTEGRATION → 增大 mask_dilation      │
│  ACTION_MISMATCH → 重写 action prompt │
│                                      │
│  → 回到 Execute（最多重试 N 次）      │
└──────────────────────────────────────┘
```

---

## 八、关于角色一致性的方案对比

### 方案 A: 旧方案 — 骨骼 + 深度合成（已放弃）
```
CharacterProxy(T-pose) → pyrender depth+mask → DepthCompositor → FLUX inpaint
```
- ❌ T-pose 僵硬，无法表达动作
- ❌ pyrender 渲染质量粗糙
- ❌ 额外的 3D 栈依赖（trimesh, pyrender）
- ❌ 深度合成复杂且不稳定

### 方案 B: 新方案 — BBox + VLM-guided Inpainting（采用）
```
VLM分析场景 → bbox + depth校验 → FLUX inpaint (prompt含角色锚定描述)
```
- ✅ 简单直接，FLUX 擅长理解文本描述
- ✅ mask 外区域完全保留 3D 渲染结果
- ✅ 动作灵活（任何姿态都可文字描述）
- ✅ 减少依赖，降低失败点
- ⚠️ 角色一致性靠 prompt 锚定 + Critic 验证循环

### 方案 C: MUSE 的方案 — 全身参考图 + Layout-guided 生成
```
生成多视角参考图 → RMBG去背 → Layout bbox → 条件生成（IP-Adapter/ControlNet）
```
- ✅ 身份一致性最强（有视觉条件）
- ❌ 需要 IP-Adapter 或 ControlNet，增加模型依赖
- ❌ 与我们"先渲染背景再注入人物"的流程冲突
- ⚠️ 作为未来升级路径保留

**最终选择：方案 B，并预留方案 C 的升级接口**

---

## 九、场景一致性四重保障

### 9.1 空间一致性（硬约束 — 我们的结构性优势）
- **同一 3DGS 模型**：所有帧从同一场景渲染，背景像素级一致
- **深度图约束**：人物 bbox 必须落在合理深度区域
- **Inpainting 保护**：mask 外区域 100% 是 3D 渲染原图

### 9.2 身份一致性（MUSE 思想）
- **参考图冻结**：Phase 0 生成全身参考图后不再修改
- **Prompt 锚定**：appearance_prompt 每帧完全相同
- **Critic 验证**：VLM 对比生成角色与参考图的相似度
- **失败重试**：身份偏移时强化 prompt 后重新生成

### 9.3 风格一致性（MUSE 思想）
- **Style Anchor**：全局风格字符串作为每个 prompt 的固定前缀
- **Style Library**：预定义风格模板，LLM 选择而非自由发挥
- **Negative Prompt**：固定的反向约束防止风格漂移

### 9.4 时序一致性
- **Sliding Window Context**：每帧的 LLM 决策都包含最近 N 帧的摘要
- **SceneState**：物品变化通过 inpainting 覆盖层持久化
- **Camera 连贯性**：参考前序帧的相机位避免突兀跳转

---

## 十、LLM 配置

首选模型：**Gemini nanobanana**（视觉能力最强）

| 调用点 | 角色 | Temperature | 是否需要视觉 |
|--------|------|-------------|-------------|
| ScreenwriterAgent | 故事→分镜 | 0.7 | 否 |
| CastingAgent.revise | 修正角色描述 | 0.3 | 否 |
| DirectorAgent | 选定视角 | 0.3 | **是（预览图）** |
| ComposerAgent.plan_layout | 人物位置规划 | 0.2 | **是（背景图）** |
| CriticAgent | 验证 | 0.2 | **是（合成图+参考图）** |
| StyleAnchor | 风格选择 | 0.5 | 否 |

所有调用通过 `LLMClient` 统一走 Gemini nanobanana API。

---

## 十一、Pydantic 数据协议

```python
# ── story_config.py ──

class CharacterDef(BaseModel):
    character_id: str
    name: str
    appearance: str          # 详细外貌描述
    personality: str         # 性格（影响动作描述）
    age: str

class CharacterAsset(BaseModel):
    """Phase 0 冻结的角色资产"""
    character_id: str
    name: str
    appearance_prompt: str       # 冻结的外貌 prompt
    reference_image_path: str    # 全身参考图路径
    style_modifier: str          # 风格前缀

class FrameSpec(BaseModel):
    """单帧规格（编剧输出）"""
    frame_id: int
    narrative: str
    camera_hint: str             # "远景", "近景", "仰拍" 等
    characters_in_frame: List[CharacterAction]
    scene_changes: List[SceneChange]
    atmosphere: str

class CharacterAction(BaseModel):
    character_id: str
    action: str                  # 动作描述
    position_hint: str           # "左侧", "中央", "远处" 等
    emotion: str

class Placement(BaseModel):
    character_id: str
    bbox: Tuple[float, float, float, float]  # (x1,y1,x2,y2) 归一化
    depth_at_feet: float
    action_prompt: str
    scale: float                 # 人物缩放比

class FrameCritique(BaseModel):
    passed: bool
    violations: List[Violation]
    revision_hints: List[str]

class Violation(BaseModel):
    type: str        # IDENTITY_DRIFT | SPATIAL_ERROR | STYLE_MISMATCH |
                     # INTEGRATION_ARTIFACT | ACTION_MISMATCH
    severity: str    # critical | minor
    detail: str
    fix_hint: str
```

---

## 十二、可替换性设计

```python
class SceneBackend(ABC):
    """场景后端接口 — 后续替换 DreamScene360 时只改这一层"""
    
    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None: ...
    
    @abstractmethod
    def get_available_cameras(self) -> List[CameraInfo]: ...
    
    @abstractmethod
    def render(self, camera_params: dict) -> RenderResult: ...

class RenderResult(BaseModel):
    rgb: np.ndarray          # [H, W, 3] float32, 0~1
    depth_raw: np.ndarray    # [H, W] float32, metric
    camera_params: dict      # 可序列化的相机参数

# DreamScene360 实现
class DreamScene360Backend(SceneBackend):
    """当前实现 — 包装 SceneRenderer"""
    ...

# 未来替换
class NewSceneBackend(SceneBackend):
    """替换时只需实现这个类"""
    ...
```

---

## 十三、实现优先级与执行路线

### P0 — 最小闭环管线（验证可行性）
1. 实现 `LLMClient._call_api()` — Gemini nanobanana 后端
2. 实现 `ScreenwriterAgent` — 故事→分镜
3. 实现 `CastingAgent` — 角色资产生成（FLUX 生成参考图）
4. 实现 `StyleAnchor` — 风格锚定
5. 实现 `DirectorAgent` — 视角选择
6. 实现 `ComposerAgent` — bbox + inpainting 合成
7. 实现 `CriticAgent` — VLM 验证
8. 实现 `StoryOrchestrator` — 串联 Phase 0 + Phase 1
9. **端到端测试: 用户输入一句话 → 输出 6 帧有人物的故事**

### P1 — 一致性强化
10. 完善 `SceneModifier` — 动态场景变化
11. 完善 `SharedMemory` — 完整状态维护
12. 添加 Post-production 全局一致性检查
13. 调参：Critic 的检查维度权重、重试策略

### P2 — 升级路径
14. IP-Adapter 集成（方案 C 的角色条件注入）
15. SceneBackend 抽象层 + 新后端对接
16. 评估指标体系（参考 MUSEBench）
