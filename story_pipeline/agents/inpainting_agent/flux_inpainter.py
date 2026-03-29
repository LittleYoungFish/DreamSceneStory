"""
FluxInpainter
=============
使用本地 FLUX.1-Fill-dev 模型，在背景图的 Mask 区域生成故事人物。

架构：
  - 基础模型  : black-forest-labs/FLUX.1-Fill-dev（已下载到 HF 缓存）
  - 技术路线  : FluxFillPipeline —— 原生 Inpainting（背景不变，Mask内重生成）
  - GPU 策略  : device_map="balanced" 均匀分布到多卡（8 × 24GB 完全够用）
  - Conda 环境: dreamstory（diffusers 0.32.2 + transformers 4.47.1 + torch 2.2.2）

运行此脚本必须使用 dreamstory 环境（不能用 dreamscene360 env）：
    conda run -n dreamstory python story_pipeline/test_step3.py

输入（来自 Step 2 输出）：
    bg_rgb.png     —— 场景 RGB 背景图（512×512 或更大）
    mask.png       —— 人物区域 Mask（白色=重绘，黑色=保留）

输出：
    frame_{id:04d}_composed.png —— 合成帧（背景 + AI 生成人物）
"""

from __future__ import annotations

import os
import gc
import numpy as np
import torch
from PIL import Image, ImageFilter
from typing import Optional

# FLUX.1-Fill-dev 的本地 HF 缓存路径或 hub ID
_FLUX_FILL_MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"


def _load_pipeline(model_id: str = _FLUX_FILL_MODEL_ID,
                   device_map: str = "auto",
                   max_memory: Optional[dict] = None) -> object:
    """
    加载 FluxFillPipeline。

    Parameters
    ----------
    model_id   : HuggingFace hub ID 或本地路径
    device_map : "balanced" / "auto" / "sequential" 等 accelerate 策略
    max_memory : 每张卡的最大显存限制，例如：
                 {0: "0GiB", 1: "22GiB", 2: "22GiB"}  ← 排除 GPU 0
                 留 None 则不限制（由 device_map 自动分配）
    """
    from diffusers import FluxFillPipeline

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    print(f"[FluxInpainter] 加载模型: {model_id}  "
          f"(device_map={device_map}, max_memory={max_memory})")

    load_kwargs: dict = dict(
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    pipe = FluxFillPipeline.from_pretrained(model_id, **load_kwargs)
    # 对 VAE 使用 tiling，避免高分辨率 OOM
    pipe.vae.enable_tiling()
    return pipe


class FluxInpainter:
    """
    对单帧执行 FLUX Fill Inpainting。

    Parameters
    ----------
    model_id : str
        HuggingFace hub ID 或本地路径，默认使用已缓存的 FLUX.1-Fill-dev。
    device_map : str
        多卡分配策略，默认 "balanced"（均匀分到所有 GPU）。
    max_memory : dict, optional
        每张卡的显存上限，用于排除被其他模型占用的 GPU。
        示例（将 FLUX 限制在 GPU 1-7，避开 SceneRenderer 所在的 GPU 0）：
            {0: "0GiB", 1: "22GiB", 2: "22GiB", ..., 7: "22GiB"}
        留 None 则让 device_map 自动决定。
    num_inference_steps : int
        推理步数，默认 28（FLUX Fill 官方推荐）。
    guidance_scale : float
        引导强度，默认 30（FLUX Fill 官方推荐用较高值）。
    """

    def __init__(
        self,
        model_id:             str            = _FLUX_FILL_MODEL_ID,
        device_map:           str            = "balanced",
        max_memory:           Optional[dict] = None,
        num_inference_steps:  int            = 28,
        guidance_scale:       float          = 30.0,
    ):
        self.model_id            = model_id
        self.device_map          = device_map
        self.max_memory          = max_memory
        self.num_inference_steps = num_inference_steps
        self.guidance_scale      = guidance_scale
        self._pipe               = None   # 懒加载

    # ──────────────────────────────────────────────────────────
    #  主接口
    # ──────────────────────────────────────────────────────────

    def inpaint(
        self,
        bg_rgb:           np.ndarray,
        mask:             np.ndarray,
        prompt:           str,
        negative_prompt:  str  = "",
        seed:             int  = 42,
        mask_dilation_px: int  = 8,
    ) -> np.ndarray:
        """
        在 bg_rgb 的 mask 区域生成人物，保留其余背景。

        Parameters
        ----------
        bg_rgb           : [H, W, 3] float32 0~1，来自 SceneRenderer.render()
        mask             : [H, W] bool，True = 重绘区域（人物）
        prompt           : 角色描述，例如 "a young woman in a red coat, standing"
        negative_prompt  : 负面提示词
        seed             : 随机种子
        mask_dilation_px : Mask 向外扩张像素数（避免边缘硬切）

        Returns
        -------
        np.ndarray [H, W, 3] float32 0~1
            合成帧（背景 + AI 生成人物）
        """
        # ── 懒加载流水线 ──────────────────────────────────────
        if self._pipe is None:
            self._pipe = _load_pipeline(
                self.model_id, self.device_map, self.max_memory
            )

        H, W = bg_rgb.shape[:2]

        # ── 图像预处理 ───────────────────────────────────────
        bg_pil   = Image.fromarray((np.clip(bg_rgb, 0, 1) * 255).astype(np.uint8))
        mask_pil = self._prepare_mask(mask, W, H, mask_dilation_px)

        # ── 推理 ──────────────────────────────────────────────
        generator = torch.Generator("cpu").manual_seed(seed)

        result = self._pipe(
            prompt             = prompt,
            image              = bg_pil,
            mask_image         = mask_pil,
            height             = H,
            width              = W,
            num_inference_steps= self.num_inference_steps,
            guidance_scale     = self.guidance_scale,
            generator          = generator,
        )

        out_pil: Image.Image = result.images[0]
        out_arr = np.array(out_pil).astype(np.float32) / 255.0
        return out_arr

    def unload(self):
        """释放 GPU 显存（下一帧推理前不调用也可，只有显存压力才需要）。"""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            gc.collect()
            torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────
    #  内部工具
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_mask(mask: np.ndarray,
                       W: int, H: int,
                       dilation_px: int) -> Image.Image:
        """
        将 bool mask 转为 PIL 灰度图（白=重绘），并可选扩张边缘。
        """
        mask_uint8 = (mask.astype(np.uint8) * 255)
        mask_pil   = Image.fromarray(mask_uint8, mode="L")

        if dilation_px > 0:
            # 用 MaxFilter 模拟 morphological dilation
            mask_pil = mask_pil.filter(
                ImageFilter.MaxFilter(size=dilation_px * 2 + 1))

        return mask_pil.resize((W, H), Image.NEAREST)

    def save_result(self,
                    result_rgb: np.ndarray,
                    output_dir: str,
                    frame_id:   int = 0) -> str:
        """保存合成帧到 output_dir/frame_{id:04d}_composed.png。"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"frame_{frame_id:04d}_composed.png")
        rgb_uint8 = (np.clip(result_rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb_uint8).save(path)
        return path
