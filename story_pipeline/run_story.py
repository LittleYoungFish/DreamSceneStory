"""
story_pipeline/run_story.py
============================
CLI 入口 — 场景一致的多智能体故事生成。

用法
----
# 最小运行（仅 LLM 编剧 + 选角，不渲染不合成）
conda run -n storyagent python story_pipeline/run_story.py \\
    --story "一个女孩在雨中的小巷里偶遇了一只猫" \\
    --num-frames 4

# 完整管线（3D 渲染 + FLUX 合成，8×4090 多卡分配）
# SceneRenderer 固定在 cuda:0，FLUX 分散到 cuda:1~7（每卡 22GiB）
conda run -n dreamscene360 python story_pipeline/run_story.py \\
    --story "一个女孩在雨中的小巷里偶遇了一只猫" \\
    --scene-model output/room_test \\
    --num-frames 6 \\
    --style cinematic_realistic \\
    --renderer-gpu 0 \\
    --flux-gpus 1,2,3,4,5,6,7 \\
    --flux-mem-per-gpu 22

# 从配置文件运行
conda run -n storyagent python story_pipeline/run_story.py \\
    --config story_pipeline/example_input.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from story_pipeline.agents.llm_client import LLMClient
from story_pipeline.orchestrator import StoryOrchestrator
from story_pipeline.story_config import CharacterDef, StoryInput


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="场景一致的多智能体故事生成系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--story", type=str, default="",
                   help="故事描述文本（与 --config 二选一）")
    p.add_argument("--config", type=str, default="",
                   help="JSON 配置文件路径（StoryInput 格式）")
    p.add_argument("--scene-model", type=str, default="",
                   help="3D 场景模型路径（如 output/room_test）")
    p.add_argument("--scene-iteration", type=int, default=10000,
                   help="3DGS checkpoint 迭代次数")
    p.add_argument("--num-frames", type=int, default=6,
                   help="期望生成的帧数")
    p.add_argument("--style", type=str, default="",
                   help="风格偏好（cinematic_realistic / anime_illustration / watercolor / ghibli / oil_painting）")
    p.add_argument("--output-dir", type=str,
                   default="story_pipeline/outputs/story_run",
                   help="输出目录")
    p.add_argument("--max-retries", type=int, default=3,
                   help="每帧最大重试次数")
    p.add_argument("--backend", type=str, default="",
                   help="LLM 后端（gemini / openai），默认读 .env")
    p.add_argument("--model", type=str, default="",
                   help="LLM 模型名称，默认读 .env")
    p.add_argument("--base-url", type=str, default="",
                   help="自定义 API base_url，例如本地代理或第三方中转（默认读 LLM_BASE_URL 环境变量）")
    p.add_argument("--renderer-gpu", type=int, default=0,
                   help="SceneRenderer (3DGS) 使用的 GPU 编号（默认 0）")
    p.add_argument("--flux-gpus", type=str, default="",
                   help="FLUX inpainter 使用的 GPU 编号，逗号分隔，例如 '1,2,3,4,5,6,7'。"
                        "留空则自动排除 --renderer-gpu 所在卡，其余卡全部给 FLUX")
    p.add_argument("--flux-mem-per-gpu", type=int, default=22,
                   help="FLUX 每张卡的显存上限 (GiB)，默认 22（4090 24GiB 留 2GiB 余量）")
    p.add_argument("--no-render", action="store_true",
                   help="跳过 3D 渲染（仅运行编剧 + 选角）")
    p.add_argument("--no-inpaint", action="store_true",
                   help="跳过 FLUX 合成（仅运行编剧 + 选角 + 渲染背景）")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="详细日志输出")
    return p.parse_args()


def _build_flux_max_memory(renderer_gpu: int,
                            flux_gpus: str,
                            mem_per_gpu: int) -> dict:
    """
    构建传给 FluxInpainter 的 max_memory 字典。

    策略：
    - renderer_gpu 分配 0GiB（禁止 FLUX 使用）
    - flux_gpus 若为空，自动检测系统 GPU 数量并排除 renderer_gpu
    - 其余每张卡分配 mem_per_gpu GiB
    """
    import torch
    total_gpus = torch.cuda.device_count()
    if total_gpus <= 1:
        # 单卡无法分离，不设限制
        return None

    if flux_gpus:
        flux_ids = [int(x.strip()) for x in flux_gpus.split(",") if x.strip()]
    else:
        flux_ids = [i for i in range(total_gpus) if i != renderer_gpu]

    if not flux_ids:
        return None

    max_memory: dict = {}
    # 禁止 FLUX 占用 renderer 所在卡
    max_memory[renderer_gpu] = "0GiB"
    for gid in flux_ids:
        max_memory[gid] = f"{mem_per_gpu}GiB"

    return max_memory


def build_story_input(args: argparse.Namespace) -> StoryInput:
    """从命令行参数或 JSON 配置构建 StoryInput。"""
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            data = json.load(f)
        return StoryInput(**data)

    if not args.story:
        print("错误: 必须指定 --story 或 --config")
        sys.exit(1)

    return StoryInput(
        story_text=args.story,
        scene_model_path=args.scene_model or "",
        scene_iteration=args.scene_iteration,
        num_frames=args.num_frames,
        style_preference=args.style,
        output_dir=args.output_dir,
        max_retries=args.max_retries,
    )


def main():
    args = parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    story_input = build_story_input(args)

    # LLM 客户端
    llm_client = LLMClient(
        backend=args.backend or None,
        model=args.model or None,
        base_url=args.base_url or None,
    )

    # 可选：3D 渲染器
    renderer = None
    if story_input.scene_model_path and not args.no_render:
        try:
            from story_pipeline.agents.layout_agent.scene_renderer import SceneRenderer
            renderer = SceneRenderer(
                model_path=story_input.scene_model_path,
                iteration=story_input.scene_iteration,
            )
            logging.getLogger("story_pipeline").info(
                f"SceneRenderer 已加载: {story_input.scene_model_path}"
            )
        except Exception as e:
            logging.getLogger("story_pipeline").warning(
                f"SceneRenderer 加载失败 ({e})，将跳过渲染步骤"
            )

    # 可选：FLUX Inpainter
    inpainter = None
    if not args.no_inpaint and renderer is not None:
        try:
            from story_pipeline.agents.inpainting_agent.flux_inpainter import FluxInpainter
            flux_max_memory = _build_flux_max_memory(
                renderer_gpu=args.renderer_gpu,
                flux_gpus=args.flux_gpus,
                mem_per_gpu=args.flux_mem_per_gpu,
            )
            log = logging.getLogger("story_pipeline")
            if flux_max_memory:
                log.info(
                    f"FLUX GPU 分配: renderer=cuda:{args.renderer_gpu}, "
                    f"flux={flux_max_memory}"
                )
            inpainter = FluxInpainter(max_memory=flux_max_memory)
            log.info("FluxInpainter 已加载")
        except Exception as e:
            logging.getLogger("story_pipeline").warning(
                f"FluxInpainter 加载失败 ({e})，将跳过合成步骤"
            )

    # 运行
    orchestrator = StoryOrchestrator(
        story_input=story_input,
        llm_client=llm_client,
        renderer=renderer,
        inpainter=inpainter,
    )

    output = orchestrator.run()

    # 打印摘要
    print("\n" + "═" * 60)
    print(f"故事: {output.title}")
    print(f"风格: {output.style.name if output.style else 'N/A'}")
    print(f"角色: {', '.join(c.name for c in output.characters)}")
    print(f"帧数: {len(output.frames)}")
    print(f"输出: {output.output_dir}")
    print("═" * 60)

    # 保存输出元信息
    meta_path = os.path.join(output.output_dir, "story_output.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, ensure_ascii=False, indent=2, default=str)
    print(f"元信息已保存: {meta_path}")


if __name__ == "__main__":
    main()
