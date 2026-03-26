import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R
from scipy.interpolate import interp1d
import copy  

def get_circular_cameras(base_cam, num_frames=300, elevation_deg=0.0):
    """Generate cameras that spin 360 degrees horizontally around the scene.

    This scene uses a coordinate system where:
      - World +Z = UP (sky), confirmed: cameras looking at ground have forward.Z = -0.795
      - Horizontal plane = XY plane
      - Camera.R stores the **c2w** rotation (getWorld2View2 does R.T to recover w2c)

    Strategy: build a look-at c2w matrix for each frame:
      forward = [cos(yaw)*cos(e), sin(yaw)*cos(e), -sin(e)]  (XY-plane orbit, tilted down)
      right   = normalize(forward × world_up)                 (always in XY plane)
      down    = normalize(forward × right)                    (camera Y = image-down)
      c2w     = [right | down | forward]  as columns
      Camera.R = c2w  (directly)

    elevation_deg: degrees below horizontal. Training horizontal cameras have ≈ 11°.
    """
    import math
    from scene.cameras import Camera
    import copy

    spinning_cams = []

    # World up axis: +Z  (confirmed: cameras looking down at ground have forward.Z = -0.795,
    # cameras looking at sky have forward.Z = +0.795)
    world_up = np.array([0.0, 0.0, 1.0])

    elev = math.radians(elevation_deg)

    # Extract initial yaw so the video starts facing the same direction as base_cam.
    # Camera.R = c2w  →  c2w[:, 2] = forward direction in world
    base_fwd_world = base_cam.R[:, 2]
    # Project onto horizontal XY plane (remove Z component)
    base_fwd_horiz = base_fwd_world.copy()
    base_fwd_horiz[2] = 0.0
    if np.linalg.norm(base_fwd_horiz) < 1e-6:
        base_fwd_horiz = np.array([1.0, 0.0, 0.0])
    else:
        base_fwd_horiz /= np.linalg.norm(base_fwd_horiz)

    # Yaw in XY plane: angle from +X axis
    initial_yaw = math.atan2(base_fwd_horiz[1], base_fwd_horiz[0])

    for i in range(num_frames):
        yaw = initial_yaw + (i / num_frames) * 2 * math.pi

        # Horizontal forward direction (in XY plane)
        fwd_horiz = np.array([math.cos(yaw), math.sin(yaw), 0.0])

        # Apply elevation: tilt forward vector toward world -X (down)
        fwd = math.cos(elev) * fwd_horiz + math.sin(elev) * (-world_up)
        fwd /= np.linalg.norm(fwd)

        # Right = fwd × world_up  (then normalise)
        right = np.cross(fwd, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0.0, 1.0, 0.0])
        else:
            right /= np.linalg.norm(right)

        # Camera up = right × fwd  (ensures orthonormal, world-up aligned)
        cam_up = np.cross(right, fwd)
        cam_up /= np.linalg.norm(cam_up)

        # c2w matrix: columns = [right, down, forward]
        # Camera Y (down in image) = -cam_up in world
        c2w = np.stack([right, -cam_up, fwd], axis=1)  # shape (3,3), c2w rotation

        # Camera.R = c2w  (getWorld2View2 does R.T internally to recover w2c)
        dummy_img = torch.zeros((3, base_cam.image_height, base_cam.image_width), dtype=torch.float32)
        created_cam = Camera(
            colmap_id=i, R=c2w, T=np.array([0.0, 0.0, 0.0]),
            FoVx=base_cam.FoVx, FoVy=base_cam.FoVy,
            image=dummy_img, gt_alpha_mask=None, image_name=f"orbit_{i}", uid=i, data_device="cuda"
        )
        spinning_cams.append(created_cam)

    return spinning_cams

def get_interpolated_cameras(cameras, num_frames=300):
    from scipy.spatial.transform import Slerp, Rotation as R
    from scipy.interpolate import interp1d
    import copy
    
    # Sort them into a continuous ring first
    sorted_cameras = get_sorted_cameras_by_yaw(cameras)
    
    # Add the first one to the end to close the loop!
    sorted_cameras.append(sorted_cameras[0])
    
    quaternions = []
    
    for cam in sorted_cameras:
        rot = R.from_matrix(cam.R)
        quaternions.append(rot.as_quat())
        
    quaternions = np.stack(quaternions)
    
    # Original keyframe indices
    times = np.linspace(0, 1, len(sorted_cameras))
    
    # Target frame indices
    new_times = np.linspace(0, 1, num_frames, endpoint=False) # Exclude very end to loop perfectly
    
    # Interpolate rotation (Slerp)
    slerp = Slerp(times, R.from_quat(quaternions))
    new_rotations = slerp(new_times).as_matrix()
    
    # Create new cameras
    interpolated_cams = []
    base_cam = sorted_cameras[0] 
    
    for i in range(num_frames):
        from scene.cameras import Camera
        new_cam = copy.copy(base_cam)
        new_cam.R = new_rotations[i]
        new_cam.T = sorted_cameras[0].T # Keep it exactly at zero perfectly
        
        dummy_img = torch.zeros((3, base_cam.image_height, base_cam.image_width), dtype=torch.float32)
        created_cam = Camera(
            colmap_id=i, R=new_cam.R, T=new_cam.T, FoVx=base_cam.FoVx, FoVy=base_cam.FoVy,
            image=dummy_img, gt_alpha_mask=None, image_name=f"interp_spin_{i}", uid=i, data_device="cuda"
        )
        interpolated_cams.append(created_cam)
        
    return interpolated_cams

def render_smooth_video(dataset: ModelParams, iteration: int, pipeline: PipelineParams, num_frames: int, elevation_deg: float = 0.0):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, api_key=None, self_refinement=None, num_prompt=None, max_rounds=None)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        train_cameras = scene.getTrainCameras()
        base_cam = train_cameras[0]
        
        print(f"Generating a 360-degree horizontal rotation path with {num_frames} frames, elevation={elevation_deg}°.")
        interp_cameras = get_circular_cameras(base_cam, num_frames=num_frames, elevation_deg=elevation_deg)

        render_path = os.path.join(dataset.model_path, "smooth_video_trajectory")
        makedirs(render_path, exist_ok=True)

        for idx, view in enumerate(tqdm(interp_cameras, desc="Rendering smooth frames")):
            render_pkg = render(view, gaussians, pipeline, background)
            rendering = render_pkg["render"]
            # 上下翻转180度 (Rotate 180 degrees to fix upside-down)
            #rendering = torch.flip(rendering, [1, 2])
            #rendering = torch.rot90(rendering, k=-1, dims=[1, 2])
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            
        print(f"Rendered frames saved to {render_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Render smooth trajectory")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_frames", default=600, type=int)
    parser.add_argument("--elevation", default=0.0, type=float,
                        help="Camera elevation in degrees. 0=horizontal, positive=tilt down toward scene.")
    
    args = get_combined_args(parser)
    safe_state(False)

    render_smooth_video(model.extract(args), args.iteration, pipeline.extract(args), args.num_frames, args.elevation)