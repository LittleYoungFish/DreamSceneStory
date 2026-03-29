"""
SceneRenderer: 对 DreamScene360 渲染 API 的最小封装。

职责：
  - 直接加载训练好的 GaussianModel（绕过 Scene.__init__ 的全套初始化流程）
  - 从 cameras.json 构造任意 Camera 对象
  - 调用 DreamScene360 原生 render() 函数，输出 RGB + 深度图

数据流：
  cameras.json  ──► build_camera_from_json()
                           │
  point_cloud.ply ──► GaussianModel.load_ply()
                           │
                    render(camera, gaussians) ──► {rgb, depth_raw, depth_vis}
"""

import os
import sys
import json
import math
import numpy as np
import torch
from types import SimpleNamespace
from PIL import Image

# ── 把 DreamScene360 根目录加入 sys.path，确保能 import 它的模块 ──
_DREAMSCENE360_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _DREAMSCENE360_ROOT not in sys.path:
    sys.path.insert(0, _DREAMSCENE360_ROOT)

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera


def _focal2fov(focal: float, pixels: int) -> float:
    """焦距 → 视角（弧度），与 DreamScene360 utils/graphics_utils.py 保持一致。"""
    return 2 * math.atan(pixels / (2 * focal))


def _build_pipeline_params() -> SimpleNamespace:
    """创建最小化的 PipelineParams，和原生 PipelineParams 等价。"""
    return SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )


class SceneRenderer:
    """
    封装 DreamScene360 的渲染管线，提供故事生成流程所需的干净接口。

    用法::

        renderer = SceneRenderer(model_path="output/room_test", iteration=10000)
        camera   = renderer.build_camera_from_json(cam_id=12)
        result   = renderer.render(camera)
        # result["rgb"]       : np.ndarray [H, W, 3]  float32  0~1
        # result["depth_raw"] : np.ndarray [H, W]     float32  原始深度（场景单位）
        # result["depth_vis"] : PIL.Image               jet colormap 可视化
    """

    def __init__(self, model_path: str, iteration: int = 10000):
        """
        Parameters
        ----------
        model_path : str
            训练输出目录，例如 ``"output/room_test"``。
            可以是相对于 DreamScene360 根目录的路径，或绝对路径。
        iteration : int
            加载哪个 checkpoint 的 point_cloud，默认 10000。
        """
        # 统一转为绝对路径
        if not os.path.isabs(model_path):
            model_path = os.path.join(_DREAMSCENE360_ROOT, model_path)

        self.model_path = model_path
        self.iteration  = iteration

        # ── 加载 cameras.json ──
        cameras_json_path = os.path.join(model_path, "cameras.json")
        if not os.path.exists(cameras_json_path):
            raise FileNotFoundError(f"cameras.json not found at: {cameras_json_path}")
        with open(cameras_json_path) as f:
            self.cameras_json = json.load(f)

        # ── 直接加载 GaussianModel，绕过 Scene 的完整初始化 ──
        ply_path = os.path.join(model_path, "point_cloud",
                                f"iteration_{iteration}", "point_cloud.ply")
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"point_cloud.ply not found at: {ply_path}")

        self.gaussians = GaussianModel(sh_degree=3)
        self.gaussians.load_ply(ply_path)
        print(f"[SceneRenderer] Loaded {self.gaussians.get_xyz.shape[0]:,} Gaussians "
              f"from iteration {iteration}.")

        self._pipeline = _build_pipeline_params()
        self._bg_black = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        self._bg_white = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    # ──────────────────────────────────────────────
    #  相机构造
    # ──────────────────────────────────────────────

    def build_camera_from_json(self, cam_id: int,
                                image_size: tuple = None) -> Camera:
        """
        从 cameras.json 的第 ``cam_id`` 条记录构造 Camera 对象。

        Parameters
        ----------
        cam_id : int
            cameras.json 中的索引（0-based），共 240 条。
        image_size : tuple (W, H), optional
            覆盖原始分辨率，默认使用 cameras.json 中的 512×512。

        Returns
        -------
        Camera
            可直接传入 render() 的相机对象。
        """
        if cam_id < 0 or cam_id >= len(self.cameras_json):
            raise IndexError(f"cam_id {cam_id} out of range [0, {len(self.cameras_json)-1}]")

        cam_data = self.cameras_json[cam_id]

        W = image_size[0] if image_size else cam_data["width"]
        H = image_size[1] if image_size else cam_data["height"]

        # cameras.json 中 rotation 是 W2C 旋转矩阵（3×3）
        R = np.array(cam_data["rotation"], dtype=np.float64)   # [3, 3]  W2C rotation

        # position 是相机中心在世界坐标系的坐标
        C = np.array(cam_data["position"], dtype=np.float64)   # [3]     camera center
        T = -R @ C                                              # [3]     W2C translation

        FoVx = _focal2fov(cam_data["fx"], W)
        FoVy = _focal2fov(cam_data["fy"], H)

        dummy_img = torch.zeros((3, H, W), dtype=torch.float32)
        return Camera(
            colmap_id=cam_id,
            R=R, T=T,
            FoVx=FoVx, FoVy=FoVy,
            image=dummy_img,
            gt_alpha_mask=None,
            image_name=cam_data.get("img_name", f"cam_{cam_id}"),
            uid=cam_id,
            data_device="cuda",
        )

    def build_camera_custom(self,
                             position: list,
                             look_at:  list,
                             up:       list = None,
                             fov_deg:  float = 60.0,
                             W: int = 512, H: int = 512) -> Camera:
        """
        根据直觉参数构造自定义相机（用于故事中不在原始 cameras.json 里的视角）。

        Parameters
        ----------
        position : [x, y, z]
            相机在世界坐标系中的位置。
        look_at : [x, y, z]
            相机注视点。
        up : [x, y, z], optional
            上方向，默认 [0, 1, 0]。
        fov_deg : float
            水平视角（度），默认 60°。
        W, H : int
            图像分辨率。
        """
        if up is None:
            up = [0.0, 1.0, 0.0]

        pos     = np.array(position, dtype=np.float64)
        target  = np.array(look_at,  dtype=np.float64)
        up_vec  = np.array(up,       dtype=np.float64)

        # 构建正交基（OpenCV 坐标系：Z 朝前，Y 朝下）
        z_axis = target - pos
        z_axis = z_axis / np.linalg.norm(z_axis)

        x_axis = np.cross(z_axis, up_vec)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)

        y_axis = np.cross(z_axis, x_axis)   # 注意：y 朝下

        # C2W 旋转 → W2C 旋转
        R_c2w = np.stack([x_axis, y_axis, z_axis], axis=1)   # [3, 3]
        R_w2c = R_c2w.T

        T = -R_w2c @ pos

        FoVx = math.radians(fov_deg)
        FoVy = _focal2fov(W / (2 * math.tan(FoVx / 2)), H)

        dummy_img = torch.zeros((3, H, W), dtype=torch.float32)
        uid = 10000 + hash((tuple(position), tuple(look_at))) % 90000
        return Camera(
            colmap_id=uid, R=R_w2c, T=T,
            FoVx=FoVx, FoVy=FoVy,
            image=dummy_img,
            gt_alpha_mask=None,
            image_name="custom_cam",
            uid=uid,
            data_device="cuda",
        )

    # ──────────────────────────────────────────────
    #  渲染
    # ──────────────────────────────────────────────

    def render(self, camera: Camera,
               white_bg: bool = False) -> dict:
        """
        对指定相机渲染场景。

        Parameters
        ----------
        camera : Camera
            由 build_camera_from_json() 或 build_camera_custom() 构造的相机。
        white_bg : bool
            True → 白色背景；False → 黑色背景（默认）。

        Returns
        -------
        dict with keys:
            ``rgb``       : np.ndarray [H, W, 3]  float32  0~1
            ``depth_raw`` : np.ndarray [H, W]     float32
            ``depth_vis`` : PIL.Image              jet colormap 可视化
        """
        bg = self._bg_white if white_bg else self._bg_black

        with torch.no_grad():
            pkg = render(camera, self.gaussians, self._pipeline, bg)

        # ── RGB ──
        rgb = pkg["render"].permute(1, 2, 0).cpu().numpy()          # [H, W, 3]

        # ── 深度 ──
        depth_raw = pkg["depth"].squeeze(0).cpu().numpy()           # [H, W]

        # ── 深度可视化（jet colormap，与 render.py 保持一致）──
        depth_vis = self._depth_to_colormap(depth_raw)

        return {
            "rgb":       rgb,
            "depth_raw": depth_raw,
            "depth_vis": depth_vis,
        }

    # ──────────────────────────────────────────────
    #  工具方法
    # ──────────────────────────────────────────────

    @staticmethod
    def _depth_to_colormap(depth: np.ndarray) -> Image.Image:
        """将原始深度图转为 jet colormap 的 PIL.Image（与 render.py 完全一致）。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        max_val = depth.max()
        if max_val > 0:
            depth_norm = depth / max_val
        else:
            depth_norm = depth

        colormap = plt.get_cmap("jet")
        colored  = colormap(depth_norm)[:, :, :3]   # [H, W, 3]  RGBA → RGB
        return Image.fromarray((colored * 255).astype(np.uint8))

    def save_render(self, result: dict, output_dir: str,
                    frame_id: int = 0) -> dict:
        """
        把 render() 的输出保存到磁盘。

        Parameters
        ----------
        result     : render() 的返回字典
        output_dir : 保存目录
        frame_id   : 帧编号，用于文件命名

        Returns
        -------
        dict  包含三个输出文件的路径：
            ``bg_rgb_path``, ``depth_vis_path``, ``depth_npy_path``
        """
        os.makedirs(output_dir, exist_ok=True)

        bg_rgb_path    = os.path.join(output_dir, f"frame_{frame_id:04d}_bg_rgb.png")
        depth_vis_path = os.path.join(output_dir, f"frame_{frame_id:04d}_depth.png")
        depth_npy_path = os.path.join(output_dir, f"frame_{frame_id:04d}_depth.npy")

        # RGB：float32 0~1 → uint8 0~255
        rgb_uint8 = (np.clip(result["rgb"], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb_uint8).save(bg_rgb_path)

        # 深度可视化
        result["depth_vis"].save(depth_vis_path)

        # 原始深度（float32 .npy，供 ControlNet 使用）
        np.save(depth_npy_path, result["depth_raw"])

        print(f"[SceneRenderer] Saved frame {frame_id}: {output_dir}")
        return {
            "bg_rgb_path":    bg_rgb_path,
            "depth_vis_path": depth_vis_path,
            "depth_npy_path": depth_npy_path,
        }

    @property
    def num_cameras(self) -> int:
        """cameras.json 中的相机总数。"""
        return len(self.cameras_json)

    def list_cameras(self, n: int = 10):
        """打印前 n 个相机的简要信息，方便选镜用。"""
        print(f"{'ID':>4}  {'img_name':<20}  {'W':>5}  {'H':>5}")
        print("-" * 40)
        for cam in self.cameras_json[:n]:
            print(f"{cam['id']:>4}  {cam['img_name']:<20}  {cam['width']:>5}  {cam['height']:>5}")
        if len(self.cameras_json) > n:
            print(f"  ... ({len(self.cameras_json)} total)")
